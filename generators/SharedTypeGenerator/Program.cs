using System.Runtime;
using CommandLine;
using NotEnoughLogs;
using SharedTypeGenerator;
using NotEnoughLogs.Behaviour;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

GCSettings.LatencyMode = GCLatencyMode.Batch;

Parser.Default.ParseArguments<GeneratorOptions>(args)
    .WithParsed(options =>
    {
        if (string.IsNullOrWhiteSpace(options.AssemblyPath))
        {
            string? discovered = ResoniteAssemblyDiscovery.TryFindRenderiteSharedDll();
            if (discovered == null)
            {
                Console.Error.WriteLine(
                    "Could not find Renderite.Shared.dll. Set RENDERITE_SHARED_DLL or RESONITE_DIR, install Resonite via Steam, or pass -i / --assembly-path.");
                Environment.Exit(1);
                return;
            }

            options.AssemblyPath = discovered;
        }
        else
        {
            options.AssemblyPath = Path.GetFullPath(options.AssemblyPath.Trim());
            if (!File.Exists(options.AssemblyPath))
            {
                Console.Error.WriteLine($"Assembly path does not exist: {options.AssemblyPath}");
                Environment.Exit(1);
                return;
            }
        }

        LogLevel maxLevel = ResolveMaxLogLevel(options.Verbose);
        using var logger = new Logger(new LoggerConfiguration
        {
            Behaviour = new DirectLoggingBehaviour(),
            MaxLevel = maxLevel,
        });
        GeneratorRunner.Run(options, logger);
    });

static LogLevel ResolveMaxLogLevel(bool verbose)
{
    if (verbose)
        return LogLevel.Trace;
#if DEBUG
    return LogLevel.Trace;
#else
    return LogLevel.Info;
#endif
}
