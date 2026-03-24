using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace Renderide.Generators.Logging;

/// <summary>
/// <see cref="ILoggerSink"/> that forwards every level except <see cref="LogLevel.Warning"/> immediately.
/// Warnings are buffered until <see cref="FlushDeferred"/> so they appear after normal progress logs
/// (errors and critical messages are not deferred so they are visible in the log as soon as they occur).
/// </summary>
public sealed class DeferringIssueSink : ILoggerSink, IDisposable
{
    private readonly ILoggerSink _inner;
    private readonly List<DeferredLogLine> _buffer = [];
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>Creates a sink that defers issue-level lines to <paramref name="inner"/> on flush.</summary>
    public DeferringIssueSink(ILoggerSink inner)
    {
        _inner = inner;
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content)
    {
        if (IsDeferredLevel(level))
        {
            lock (_lock)
                _buffer.Add(new DeferredLogLine(level, new string(category), new string(content)));
            return;
        }

        _inner.Log(level, category, content);
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args)
    {
        if (IsDeferredLevel(level))
        {
            string message = string.Format(format.ToString(), args);
            lock (_lock)
                _buffer.Add(new DeferredLogLine(level, new string(category), message));
            return;
        }

        _inner.Log(level, category, format, args);
    }

    /// <summary>Writes all buffered warning lines through the inner sink, then clears the buffer.</summary>
    public void FlushDeferred()
    {
        List<DeferredLogLine> batch;
        lock (_lock)
        {
            if (_buffer.Count == 0)
                return;
            batch = _buffer.ToList();
            _buffer.Clear();
        }

        _inner.Log(LogLevel.Info, "Issues", "");
        _inner.Log(LogLevel.Info, "Issues", "─── Deferred warnings ───");

        foreach (DeferredLogLine line in batch)
            _inner.Log(line.Level, line.Category, line.Message);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        FlushDeferred();
        if (_inner is IDisposable d)
            d.Dispose();
        _disposed = true;
    }

    private static bool IsDeferredLevel(LogLevel level) => level == LogLevel.Warning;

    private readonly record struct DeferredLogLine(LogLevel Level, string Category, string Message);
}
