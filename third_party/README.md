# Third-party vendored sources

| Path | Purpose |
|------|---------|
| `ggml/` | [ggml-org/ggml](https://github.com/ggml-org/ggml) git submodule for `cpp/` C++ inference |

Initialize after clone:

```bash
git submodule update --init third_party/ggml
```

Pinned to ggml **0.17.0** (`9be3133`) for parity with ACloudViewer / LightGlue-GGML builds.
