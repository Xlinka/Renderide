//! Orphan process cleanup from previous crashed runs.

use std::fs;
use std::io::Write;

use crate::paths;

/// Kills orphaned Host/renderer processes from a previous crashed run.
/// Call before spawning anything.
#[cfg(unix)]
pub fn kill_orphans() {
    let path = paths::pid_file_path();
    let Ok(contents) = fs::read_to_string(&path) else {
        return;
    };
    let _ = fs::remove_file(&path);

    for line in contents.lines() {
        let pid = match line
            .strip_prefix("host:")
            .or_else(|| line.strip_prefix("renderer:"))
        {
            Some(rest) => rest.trim().parse::<i32>().ok(),
            None => continue,
        };
        let Some(pid) = pid else {
            continue;
        };
        if unsafe { libc::kill(pid, 0) } == 0 {
            logger::info!("Killing orphan process {} from previous run", pid);
            let _ = unsafe { libc::kill(pid, libc::SIGTERM) };
        }
    }
}

#[cfg(not(unix))]
pub fn kill_orphans() {}

/// Appends a PID entry to the PID file.
pub fn write_pid_file(pid: u32, kind: &str) {
    let path = paths::pid_file_path();
    match fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{}:{}", kind, pid) {
                logger::error!("Failed to write PID file: {}", e);
            }
            let _ = f.flush();
        }
        Err(e) => {
            logger::error!("Failed to open PID file: {}", e);
        }
    }
}

/// Removes the PID file at shutdown.
pub fn remove_pid_file() {
    let _ = fs::remove_file(paths::pid_file_path());
}
