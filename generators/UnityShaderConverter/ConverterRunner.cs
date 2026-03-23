using System.Linq;
using NotEnoughLogs;
using UnityShaderConverter.Analysis;
using UnityShaderConverter.Config;
using UnityShaderConverter.Emission;
using UnityShaderConverter.Logging;
using UnityShaderConverter.Options;
using UnityShaderConverter.Variants;
using UnityShaderParser.Common;

namespace UnityShaderConverter;

/// <summary>Orchestrates discovery, parsing, Slang/WGSL emission, and Rust codegen into <c>crates/renderide/src/shaders/&lt;mod&gt;/</c>.</summary>
public static class ConverterRunner
{
    /// <summary>Runs the full pipeline for the given options and returns a process exit code.</summary>
    public static int Run(ConverterOptions options, Logger logger)
    {
        options.DetermineDefaultPaths();
        string outputDir = Path.GetFullPath(options.OutputDirectory ?? ".");
        Directory.CreateDirectory(outputDir);

        string renderideRoot = FindRenderideRoot(outputDir);
        logger.LogInfo(LogCategory.Startup, $"Renderide root: {renderideRoot}");
        logger.LogInfo(LogCategory.Startup, $"Shader output directory: {outputDir}");

        string legacyGenerated = Path.Combine(renderideRoot, "crates", "renderide", "src", "shaders", "generated");
        TryDeleteDirectoryRecursive(legacyGenerated, logger);

        string exeDir = AppContext.BaseDirectory;
        var defaults = DefaultCompilerConfig.LoadFromOutputDirectory(exeDir);
        var compilerConfig = ConfigLoader.MergeCompilerConfig(defaults, options.CompilerConfigPath);
        VariantConfigModel? variantConfig = ConfigLoader.LoadVariantConfig(options.VariantConfigPath);

        string runtimeSlangDir = Path.Combine(exeDir, "runtime_slang");
        if (!Directory.Exists(runtimeSlangDir))
        {
            logger.LogInfo(LogCategory.Startup, $"runtime_slang not next to executable ({runtimeSlangDir}); using source tree path.");
            runtimeSlangDir = Path.GetFullPath(Path.Combine(renderideRoot, "generators", "UnityShaderConverter", "runtime_slang"));
        }

        IReadOnlyList<string> inputs = ShaderDiscovery.Enumerate(options.InputDirectories ?? Array.Empty<string>());
        logger.LogInfo(LogCategory.Startup, $"Discovered {inputs.Count} .shader files.");

        string slangcExe = SlangCompiler.ResolveExecutable(options.SlangcPath);
        var slangCompiler = new SlangCompiler(slangcExe, logger);

        string tempSlangDir = Path.Combine(Path.GetTempPath(), "UnityShaderConverter", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempSlangDir);

        var bundleEntries = new List<ShaderBundleEntry>();
        var modNameOwners = new Dictionary<string, string>(StringComparer.Ordinal);
        int variantFailures = 0;

        try
        {
            foreach (string shaderPath in inputs)
            {
                string relForGlob = Path.GetRelativePath(renderideRoot, shaderPath).Replace('\\', '/');
                if (!ShaderLabAnalyzer.TryAnalyze(shaderPath, out ShaderFileDocument? doc, out List<Diagnostic> diags, out List<string> errors))
                {
                    foreach (string e in errors)
                        logger.LogWarning(LogCategory.Parse, $"{shaderPath}: {e}");
                    foreach (Diagnostic d in diags.Where(d => (d.Kind & DiagnosticFlags.OnlyErrors) != 0))
                        logger.LogWarning(LogCategory.Parse, d.ToString());
                    continue;
                }

                foreach (Diagnostic d in diags.Where(d => (d.Kind & DiagnosticFlags.Warning) != 0))
                    logger.LogWarning(LogCategory.Parse, d.ToString());

                if (doc!.Passes.Count == 0)
                {
                    logger.LogWarning(LogCategory.Parse, $"{shaderPath}: no code passes; skipping.");
                    continue;
                }

                IReadOnlyList<IReadOnlyList<string>> variants;
                try
                {
                    variants = VariantExpander.Expand(doc, compilerConfig, variantConfig);
                }
                catch (InvalidOperationException ex)
                {
                    logger.LogWarning(LogCategory.Variants, $"{shaderPath}: {ex.Message}");
                    variantFailures++;
                    continue;
                }

                bool slangEligible = GlobMatcher.MatchesAny(relForGlob, compilerConfig.SlangEligibleGlobPatterns);
                string shaderDir = Path.GetDirectoryName(shaderPath) ?? ".";

                string modName = RustEmitter.ModuleNameFromShaderName(doc.ShaderName);
                if (!modNameOwners.TryGetValue(modName, out string? ownerPath))
                    modNameOwners[modName] = shaderPath;
                else if (!string.Equals(Path.GetFullPath(ownerPath), Path.GetFullPath(shaderPath), StringComparison.Ordinal))
                {
                    logger.LogWarning(
                        LogCategory.Rust,
                        $"Skipping `{shaderPath}`: Rust module name `{modName}` is already used by `{ownerPath}` (duplicate Unity shader name).");
                    continue;
                }

                IReadOnlyList<SpecializationAxis> axes = SpecializationExtractor.Extract(doc, compilerConfig);
                var axisKeywords = new HashSet<string>(axes.Select(a => a.Keyword), StringComparer.Ordinal);
                List<string> firstVariantDefines = variants[0].Where(s => s.Length > 0).ToList();
                List<string> baselineDefines = firstVariantDefines.Where(d => !axisKeywords.Contains(d)).ToList();

                string modDir = Path.Combine(outputDir, modName);
                Directory.CreateDirectory(modDir);

                bool allPassesOk = true;
                bool anyPassDroppedSpecialization = false;
                for (int pi = 0; pi < doc.Passes.Count; pi++)
                {
                    ShaderPassDocument pass = doc.Passes[pi];
                    string wgslPath = Path.Combine(modDir, RustEmitter.WgslPassFileName(pi));

                    if (!options.SkipSlang && slangEligible)
                    {
                        PassCompileOutcome outcome = TryCompilePassWithOptionalFallback(
                            pass,
                            baselineDefines,
                            axes,
                            firstVariantDefines,
                            tempSlangDir,
                            wgslPath,
                            runtimeSlangDir,
                            shaderDir,
                            slangCompiler,
                            shaderPath,
                            pi,
                            logger);
                        if (outcome == PassCompileOutcome.Failed)
                            allPassesOk = false;
                        else if (outcome == PassCompileOutcome.OkWithoutSpecialization)
                            anyPassDroppedSpecialization = true;
                    }
                    else if (!File.Exists(wgslPath) || new FileInfo(wgslPath).Length == 0)
                    {
                        logger.LogWarning(
                            LogCategory.Output,
                            $"No WGSL at {wgslPath} for `{doc.ShaderName}` pass {pi}; use --skip-slang only when files exist.");
                        allPassesOk = false;
                    }
                }

                if (!allPassesOk)
                {
                    logger.LogWarning(LogCategory.Rust, $"Skipping shader module `{modName}` for `{doc.ShaderName}` (missing WGSL).");
                    TryDeleteDirectoryRecursive(modDir, logger);
                    continue;
                }

                IReadOnlyList<SpecializationAxis> rustAxes =
                    anyPassDroppedSpecialization ? Array.Empty<SpecializationAxis>() : axes;

                string sourceComment = Path.GetRelativePath(renderideRoot, doc.SourcePath);
                bundleEntries.Add(new ShaderBundleEntry(modName, doc, sourceComment.Replace('\\', '/'), doc.Passes.Count, rustAxes));
                logger.LogInfo(LogCategory.Rust, $"Queued shader `{modName}` ({doc.Passes.Count} pass(es), {axes.Count} specialization axis/axes).");
            }

            bundleEntries.Sort((a, b) => string.CompareOrdinal(a.ModName, b.ModName));
            var currentMods = bundleEntries.Select(e => e.ModName).ToHashSet(StringComparer.Ordinal);

            CleanLegacyBundleFiles(outputDir, logger);
            RemoveStaleConverterShaderDirectories(outputDir, currentMods, logger);

            foreach (ShaderBundleEntry e in bundleEntries)
            {
                string modDir = Path.Combine(outputDir, e.ModName);
                Directory.CreateDirectory(modDir);
                File.WriteAllText(Path.Combine(modDir, "mod.rs"), RustEmitter.EmitShaderCrateModRs());
                File.WriteAllText(
                    Path.Combine(modDir, "material.rs"),
                    RustEmitter.EmitShaderMaterialRs(e.Document, e.SourceComment, e.PassCount, e.Axes));
                logger.LogInfo(LogCategory.Rust, $"Wrote {modDir}");
            }
        }
        finally
        {
            TryDeleteDirectoryRecursive(tempSlangDir, logger);
        }

        logger.LogInfo(LogCategory.Output, $"Done. Shader folders: {bundleEntries.Count}, variant limit skips: {variantFailures}.");
        return 0;
    }

    private sealed record ShaderBundleEntry(
        string ModName,
        ShaderFileDocument Document,
        string SourceComment,
        int PassCount,
        IReadOnlyList<SpecializationAxis> Axes);

    private enum PassCompileOutcome
    {
        Failed,
        Ok,
        OkWithoutSpecialization,
    }

    private static PassCompileOutcome TryCompilePassWithOptionalFallback(
        ShaderPassDocument pass,
        List<string> baselineDefines,
        IReadOnlyList<SpecializationAxis> axes,
        List<string> fullFirstVariantDefines,
        string tempSlangDir,
        string wgslPath,
        string runtimeSlangDir,
        string shaderSourceIncludeDir,
        SlangCompiler slangCompiler,
        string shaderPath,
        int passIndex,
        Logger logger)
    {
        string tempSlangPath = Path.Combine(tempSlangDir, "compile.slang");
        try
        {
            if (TryCompileOnce(
                    pass,
                    baselineDefines,
                    axes,
                    tempSlangPath,
                    wgslPath,
                    runtimeSlangDir,
                    shaderSourceIncludeDir,
                    slangCompiler,
                    shaderPath,
                    passIndex,
                    logger))
                return axes.Count > 0 ? PassCompileOutcome.Ok : PassCompileOutcome.OkWithoutSpecialization;

            if (axes.Count > 0)
            {
                logger.LogWarning(
                    LogCategory.SlangCompile,
                    $"{shaderPath} pass {passIndex}: retrying without specialization injection (full first-variant defines only).");
                if (TryCompileOnce(
                        pass,
                        fullFirstVariantDefines,
                        Array.Empty<SpecializationAxis>(),
                        tempSlangPath,
                        wgslPath,
                        runtimeSlangDir,
                        shaderSourceIncludeDir,
                        slangCompiler,
                        shaderPath,
                        passIndex,
                        logger))
                    return PassCompileOutcome.OkWithoutSpecialization;
            }

            return PassCompileOutcome.Failed;
        }
        finally
        {
            TryDeleteFile(tempSlangPath);
        }
    }

    private static bool TryCompileOnce(
        ShaderPassDocument pass,
        List<string> baselineDefines,
        IReadOnlyList<SpecializationAxis> axes,
        string tempSlangPath,
        string wgslPath,
        string runtimeSlangDir,
        string shaderSourceIncludeDir,
        SlangCompiler slangCompiler,
        string shaderPath,
        int passIndex,
        Logger logger)
    {
        string slangSource = SlangEmitter.EmitPassSlang(pass, baselineDefines, axes);
        File.WriteAllText(tempSlangPath, slangSource);
        logger.LogDebug(LogCategory.Slang, $"Transient Slang → slangc ({tempSlangPath})");
        if (!slangCompiler.TryCompileToWgsl(
                tempSlangPath,
                wgslPath,
                runtimeSlangDir,
                shaderSourceIncludeDir,
                pass.VertexEntry!,
                pass.FragmentEntry!,
                baselineDefines,
                out string? err))
        {
            logger.LogWarning(LogCategory.SlangCompile, $"{shaderPath} pass {passIndex}: slangc failed: {err}");
            return false;
        }

        logger.LogInfo(LogCategory.SlangCompile, $"WGSL {wgslPath}");
        return true;
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
            // ignored
        }
    }

    private static void CleanLegacyBundleFiles(string shadersRoot, Logger logger)
    {
        foreach (string legacy in new[] { "wgsl_sources.rs", "materials.rs" })
        {
            string p = Path.Combine(shadersRoot, legacy);
            if (!File.Exists(p))
                continue;
            try
            {
                File.Delete(p);
                logger.LogInfo(LogCategory.Rust, $"Removed legacy {p}");
            }
            catch (Exception ex)
            {
                logger.LogWarning(LogCategory.Rust, $"Could not delete legacy {p}: {ex.Message}");
            }
        }
    }

    private static void RemoveStaleConverterShaderDirectories(string shadersRoot, HashSet<string> currentMods, Logger logger)
    {
        if (!Directory.Exists(shadersRoot))
            return;
        foreach (string dir in Directory.GetDirectories(shadersRoot))
        {
            string name = Path.GetFileName(dir);
            if (name is null || currentMods.Contains(name))
                continue;
            string marker = Path.Combine(dir, "mod.rs");
            if (!File.Exists(marker))
                continue;
            string head = File.ReadAllText(marker);
            if (!head.Contains("UnityShaderConverter", StringComparison.Ordinal))
                continue;
            TryDeleteDirectoryRecursive(dir, logger);
        }
    }

    private static void TryDeleteDirectoryRecursive(string path, Logger logger)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
                logger.LogInfo(LogCategory.Output, $"Removed directory {path}");
            }
        }
        catch (Exception ex)
        {
            logger.LogWarning(LogCategory.Output, $"Could not remove {path}: {ex.Message}");
        }
    }

    private static string FindRenderideRoot(string outputDirectory)
    {
        var dir = new DirectoryInfo(Path.GetFullPath(outputDirectory));
        while (dir is not null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "crates", "renderide")))
                return dir.FullName;
            dir = dir.Parent;
        }

        return Path.GetFullPath(Path.Combine(outputDirectory, "..", "..", ".."));
    }
}
