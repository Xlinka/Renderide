using System.Text;
using UnityShaderConverter.Analysis;
using UnityShaderConverter.Config;

namespace UnityShaderConverter.Variants;

/// <summary>Maps <c>#pragma multi_compile</c> / <c>shader_feature</c> keywords to Slang specialization constants.</summary>
public static class SpecializationExtractor
{
    /// <summary>Builds specialization axes (stable order: first-seen keyword order).</summary>
    public static IReadOnlyList<SpecializationAxis> Extract(ShaderFileDocument document, CompilerConfigModel compilerConfig)
    {
        if (!compilerConfig.EnableSlangSpecialization)
            return Array.Empty<SpecializationAxis>();

        var seen = new HashSet<string>(StringComparer.Ordinal);
        var axes = new List<SpecializationAxis>();
        int max = Math.Max(1, compilerConfig.MaxSpecializationConstants);
        foreach (string line in document.MultiCompilePragmas)
        {
            if (VariantExpander.IsIgnoredMultiCompilePragmaLine(line))
                continue;
            List<string>? opts = VariantExpander.ParseMultiCompileKeywords(line);
            if (opts is null)
                continue;
            foreach (string kw in opts)
            {
                if (kw.Length == 0 || seen.Contains(kw))
                    continue;
                seen.Add(kw);
                string slangId = SlangIdentifierForKeyword(kw);
                string rustField = RustFieldNameForKeyword(kw);
                axes.Add(new SpecializationAxis(axes.Count, kw, slangId, rustField));
                if (axes.Count >= max)
                    return axes;
            }
        }

        return axes;
    }

    private static string SlangIdentifierForKeyword(string keyword)
    {
        string safe = SanitizeForIdent(keyword);
        return "USC_" + safe;
    }

    private static string RustFieldNameForKeyword(string keyword)
    {
        string snake = PascalOrUpperToSnake(keyword);
        if (snake.Length == 0)
            snake = "kw";
        if (char.IsDigit(snake[0]))
            snake = "kw_" + snake;
        return snake;
    }

    private static string SanitizeForIdent(string s)
    {
        var sb = new StringBuilder();
        foreach (char c in s)
        {
            if (char.IsLetterOrDigit(c))
                sb.Append(c);
            else if (c is '_' or '-')
                sb.Append('_');
        }

        return sb.Length > 0 ? sb.ToString() : "KW";
    }

    private static string PascalOrUpperToSnake(string s)
    {
        if (s.Length == 0)
            return "";
        var sb = new StringBuilder();
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '_' || c == '-')
            {
                sb.Append('_');
                continue;
            }

            if (char.IsUpper(c) && i > 0 && s[i - 1] != '_' && !char.IsUpper(s[i - 1]))
                sb.Append('_');
            sb.Append(char.ToLowerInvariant(c));
        }

        return sb.ToString().Trim('_');
    }
}

/// <summary>One <c>vk::constant_id</c> axis for Slang → WGSL overrides.</summary>
/// <param name="ConstantId">Matches WGSL <c>@id</c> / wgpu pipeline constant key as decimal string.</param>
/// <param name="Keyword">Original ShaderLab keyword (e.g. from <c>multi_compile</c>).</param>
/// <param name="SlangIdentifier">Bool const name in emitted Slang (and typically WGSL override name).</param>
/// <param name="RustFieldName">Field on generated <c>VariantKey</c>.</param>
public sealed record SpecializationAxis(
    int ConstantId,
    string Keyword,
    string SlangIdentifier,
    string RustFieldName);
