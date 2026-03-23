using UnityShaderConverter;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for input-root resolution (WGSL paths now live next to each shader module).</summary>
public sealed class ShaderInputPathsTests
{
    [Fact]
    public void FindContainingInputRoot_Prefers_longest_matching_root()
    {
        string temp = Path.Combine(Path.GetTempPath(), "usc_test_" + Guid.NewGuid().ToString("N"));
        string rootA = Path.Combine(temp, "a");
        string rootB = Path.Combine(temp, "a", "b");
        string shader = Path.Combine(rootB, "x.shader");
        Directory.CreateDirectory(rootB);
        File.WriteAllText(shader, "Shader \"T\" { }");

        try
        {
            string? found = ShaderInputPaths.FindContainingInputRoot(shader, new[] { rootA, rootB });
            Assert.Equal(Path.GetFullPath(rootB), Path.GetFullPath(found!));
        }
        finally
        {
            try
            {
                Directory.Delete(temp, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    [Fact]
    public void GetWgslNestedDirectoryRelativeToWgslRoot_Mirrors_relative_path_without_file()
    {
        string temp = Path.GetTempPath().TrimEnd(Path.DirectorySeparatorChar);
        string root = Path.Combine(temp, "roots", "shaders");
        string shader = Path.Combine(root, "Billboard", "Unlit.shader");
        string nested = ShaderInputPaths.GetWgslNestedDirectoryRelativeToWgslRoot(shader, root);
        Assert.Equal("Billboard", nested.Replace('\\', '/'));
    }
}
