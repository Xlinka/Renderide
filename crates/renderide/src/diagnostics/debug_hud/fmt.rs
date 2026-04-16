//! Right-aligned numeric [`format!`] helpers so HUD columns keep a stable width.

/// Formats `value` as a right-aligned decimal with `decimals` places and total width `width`.
pub fn f64_field(width: usize, decimals: usize, value: f64) -> String {
    format!("{value:>w$.d$}", w = width, d = decimals)
}

/// Human-readable gibibytes from bytes (numeric part only; caller adds `GiB` suffix).
pub fn gib_value(width: usize, decimals: usize, bytes: u64) -> String {
    let g = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    f64_field(width, decimals, g)
}

/// Formats byte counts for dense allocator tables (B / KiB / MiB / GiB / TiB).
pub fn bytes_compact(bytes: u64) -> String {
    const SUFFIX: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut idx = 0usize;
    let mut amount = bytes as f64;
    while amount >= 1024.0 && idx < SUFFIX.len() - 1 {
        amount /= 1024.0;
        idx += 1;
    }
    format!("{amount:.2} {}", SUFFIX[idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_fmt_produces_stable_field_width() {
        assert_eq!(f64_field(8, 2, 1.0).len(), 8);
        assert_eq!(f64_field(8, 2, 123.456).len(), 8);
    }

    #[test]
    fn bytes_compact_zero() {
        assert_eq!(bytes_compact(0), "0.00 B");
    }
}
