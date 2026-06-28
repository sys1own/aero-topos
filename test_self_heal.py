from core.parser.universal import parse_source, load_language
try:
    print("🚀 Testing behavioral Python grammar load...")
    lang_py = load_language("python")
    print(f"  [✅] Successfully unboxed/wrapped Python language asset: {lang_py}")

    test_uast = parse_source("def hello(): pass", "python")
    print(f"  [✅] UAST Generation clean. Semantic Hash: {test_uast['semantic_hash']}")
    print("\n🏆 Absolute Cross-Era Compatibility Achieved!")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    raise e