//! Append-only panic logging so panic hooks never block on the global logger mutex.
//!
//! Prefer [`append_panic_report_to_file`] and [`log_panic`] from panic handlers; use [`log_panic_payload`]
//! only when you already have a `catch_unwind` payload and an initialized global logger.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::level::LogLevel;
use crate::output;

/// Logs a panic payload from `catch_unwind`. Extracts [`String`] or `&'static str` if possible.
///
/// Use when handling [`Err`] from [`std::panic::catch_unwind`] to surface the panic message through
/// the normal logger (requires an initialized global logger).
pub fn log_panic_payload(payload: Box<dyn std::any::Any + Send>, context: &str) {
    let msg = match payload.downcast::<String>() {
        Ok(s) => format!("{context}: {}", *s),
        Err(p) => match p.downcast::<&'static str>() {
            Ok(s) => format!("{context}: {}", *s),
            Err(_) => format!("{context}: panic (payload type not string)"),
        },
    };
    output::log(LogLevel::Error, format_args!("{msg}"));
}

/// Formats a panic line and full backtrace for logging and optional terminal output.
///
/// Uses [`std::backtrace::Backtrace::force_capture`] so backtraces are recorded regardless of
/// `RUST_BACKTRACE`.
pub fn panic_report(info: &dyn std::fmt::Display) -> String {
    format!(
        "PANIC: {info}\nBacktrace:\n{:#?}\n",
        std::backtrace::Backtrace::force_capture()
    )
}

/// Appends a pre-formatted panic report to the log file without acquiring the global logger mutex.
///
/// Safe to call from a panic hook alongside [`panic_report`].
pub fn append_panic_report_to_file(path: impl AsRef<Path>, report: &str) {
    let path = path.as_ref();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(report.as_bytes());
        let _ = f.flush();
        let _ = f.sync_all();
    }
}

/// Writes panic info and backtrace to the given log file. Flushes immediately so the panic
/// is visible on disk. Does not acquire the logger mutex (safe from a panic handler).
///
/// Uses [`panic_report`] internally.
pub fn log_panic(path: impl AsRef<Path>, info: &dyn std::fmt::Display) {
    let report = panic_report(info);
    append_panic_report_to_file(path, &report);
}
