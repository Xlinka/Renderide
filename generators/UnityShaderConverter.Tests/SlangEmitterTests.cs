using UnityShaderConverter.Analysis;
using UnityShaderConverter.Emission;
using UnityShaderConverter.Variants;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Tests;

/// <summary>Snapshot-style checks for emitted Slang.</summary>
public sealed class SlangEmitterTests
{
    /// <summary>Emitted Slang includes compat header and strips <c>#pragma vertex</c>.</summary>
    [Fact]
    public void EmitPassSlang_IncludesUnityCompatAndStripsPragmas()
    {
        var pass = new ShaderPassDocument
        {
            PassName = null,
            PassIndex = 0,
            ProgramSource = "#pragma vertex vert\n#pragma fragment frag\nfloat4 main() { return 0; }\n",
            Pragmas = Array.Empty<string>(),
            VertexEntry = "vert",
            FragmentEntry = "frag",
            RenderStateSummary = "",
            FixedFunctionState = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), new Dictionary<string, string>()),
        };
        string slang = SlangEmitter.EmitPassSlang(pass, Array.Empty<string>(), Array.Empty<SpecializationAxis>());
        Assert.Contains("#include \"UnityCompat.slang\"", slang);
        Assert.Contains("#include \"UnityCompatPostUnity.slang\"", slang);
        Assert.DoesNotContain("#pragma vertex", slang);
        Assert.DoesNotContain("#pragma fragment", slang);
    }

    /// <summary>Post-Unity compat include is inserted after the first run of <c>#include</c> lines.</summary>
    [Fact]
    public void InsertUnityCompatPostInclude_AfterInitialIncludes()
    {
        const string body = "#include \"A.cginc\"\n#include \"B.cginc\"\n\nfloat x;\n";
        string result = SlangEmitter.InsertUnityCompatPostIncludeAfterInitialIncludes(body);
        int post = result.IndexOf("UnityCompatPostUnity.slang", StringComparison.Ordinal);
        int b = result.IndexOf("#include \"B.cginc\"", StringComparison.Ordinal);
        int x = result.IndexOf("float x", StringComparison.Ordinal);
        Assert.True(post > b);
        Assert.True(x > post);
    }

    /// <summary>Specialization axes become <c>[vk::constant_id(n)]</c> bools before UnityCompat.</summary>
    [Fact]
    public void EmitPassSlang_WithAxes_InjectsVkConstantIdBools()
    {
        var pass = new ShaderPassDocument
        {
            PassName = null,
            PassIndex = 0,
            ProgramSource = "#pragma vertex vert\n#pragma fragment frag\nfloat4 main() { return 0; }\n",
            Pragmas = Array.Empty<string>(),
            VertexEntry = "vert",
            FragmentEntry = "frag",
            RenderStateSummary = "",
            FixedFunctionState = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), new Dictionary<string, string>()),
        };
        var axes = new[]
        {
            new SpecializationAxis(0, "FOO", "USC_FOO", "foo"),
            new SpecializationAxis(1, "BAR", "USC_BAR", "bar"),
        };
        string slang = SlangEmitter.EmitPassSlang(pass, Array.Empty<string>(), axes);
        Assert.Contains("[vk::constant_id(0)]", slang);
        Assert.Contains("[vk::constant_id(1)]", slang);
        Assert.Contains("const bool USC_FOO", slang);
        Assert.Contains("const bool USC_BAR", slang);
    }

    /// <summary>Preprocessor maps <c>#ifdef</c> on axis keywords to <c>USC_*</c>.</summary>
    [Fact]
    public void EmitPassSlang_RewritesIfdefToSlangAxisName()
    {
        var pass = new ShaderPassDocument
        {
            PassName = null,
            PassIndex = 0,
            ProgramSource = "#pragma vertex vert\n#pragma fragment frag\n#ifdef FOO\nfloat z = 1;\n#endif\n",
            Pragmas = Array.Empty<string>(),
            VertexEntry = "vert",
            FragmentEntry = "frag",
            RenderStateSummary = "",
            FixedFunctionState = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), new Dictionary<string, string>()),
        };
        var axes = new[] { new SpecializationAxis(0, "FOO", "USC_FOO", "foo") };
        string slang = SlangEmitter.EmitPassSlang(pass, Array.Empty<string>(), axes);
        Assert.Contains("#ifdef USC_FOO", slang);
        Assert.DoesNotContain("#ifdef FOO", slang);
    }

    /// <summary>Legacy <c>sampler2D _Tex;</c> becomes <c>UNITY_DECLARE_TEX2D</c> after PostUnity include resolution.</summary>
    [Fact]
    public void RewriteLegacySampler2DDeclarations_ReplacesUnityStyleTextures()
    {
        const string body = "  sampler2D _MainTex;\n  sampler2D _DepthTex;\n";
        string rewritten = SlangEmitter.RewriteLegacySampler2DDeclarations(body);
        Assert.Contains("UNITY_DECLARE_TEX2D(_MainTex)", rewritten, StringComparison.Ordinal);
        Assert.Contains("UNITY_DECLARE_TEX2D(_DepthTex)", rewritten, StringComparison.Ordinal);
        Assert.DoesNotContain("sampler2D _MainTex", rewritten, StringComparison.Ordinal);
    }
}
