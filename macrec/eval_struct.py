"""mmgen addition: eval_split.py (trie beam search, valid or test, cost) for StructT5 checkpoints (§16): test_ddp_save
loads the model through its module-level T5ForConditionalGeneration, which is pointed at enc_struct.StructT5 (it
rebuilds the E1 / E2 structure from config.struct_mode). Same arguments as eval_split.py.
"""
import enc_struct
import eval_split
import test_ddp_save

if __name__ == "__main__":
    test_ddp_save.T5ForConditionalGeneration = enc_struct.StructT5
    eval_split.main()
