//! Canonical essential WGSL shader programs resolved from uploaded Unity shader names.

/// Small supported shader set for the renderer's native WGSL implementations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EssentialShaderProgram {
    /// No native WGSL equivalent is registered for this shader name.
    Unsupported,
    /// Resonite `PBSMetallic` family.
    PbsMetallic,
    /// Resonite world `Shader "Unlit"`.
    WorldUnlit,
    /// Resonite `UI/Unlit`.
    UiUnlit,
    /// Resonite `UI/Text/Unlit`.
    UiTextUnlit,
}

fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Resolves the essential WGSL program from a Unity shader name or plain upload label.
pub fn resolve_essential_shader_program(name: Option<&str>) -> EssentialShaderProgram {
    let Some(name) = name else {
        return EssentialShaderProgram::Unsupported;
    };
    let Some(token) = name.split_whitespace().next() else {
        return EssentialShaderProgram::Unsupported;
    };
    let key = compact_alnum_lower(token);
    match key.as_str() {
        "pbsmetallic" => EssentialShaderProgram::PbsMetallic,
        "unlit" => EssentialShaderProgram::WorldUnlit,
        "uiunlit" => EssentialShaderProgram::UiUnlit,
        "uitextunlit" => EssentialShaderProgram::UiTextUnlit,
        _ => EssentialShaderProgram::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::{EssentialShaderProgram, resolve_essential_shader_program};

    #[test]
    fn resolves_essential_programs() {
        assert_eq!(
            resolve_essential_shader_program(Some("PBSMetallic")),
            EssentialShaderProgram::PbsMetallic
        );
        assert_eq!(
            resolve_essential_shader_program(Some("Unlit")),
            EssentialShaderProgram::WorldUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("UI_Unlit")),
            EssentialShaderProgram::UiUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("UI_TextUnlit")),
            EssentialShaderProgram::UiTextUnlit
        );
    }
}
