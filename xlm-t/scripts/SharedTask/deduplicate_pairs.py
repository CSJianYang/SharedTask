import argparse
import fileinput
import hashlib
import sys
from multiprocessing import Pool


def get_hashes_and_lines(raw_line):
    hash = hashlib.md5(raw_line.encode("utf-8")).hexdigest()
    return hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--src", "-src", type=str, default="/mnt/input/SharedTask/thunder/Bitext_v1_sup/amen/train.am-en.am", help="input files")
    parser.add_argument("--tgt", "-tgt", type=str, default="/mnt/input/SharedTask/thunder/Bitext_v1_sup/amen/train.am-en.en", help="input files")
    parser.add_argument("--new-src", "-new-src", type=str, default="/mnt/input/SharedTask/thunder/Bitext_v1_sup/amen/train.decup.am-en.am", help="input files")
    parser.add_argument("--new-tgt", "-new-tgt", type=str, default="/mnt/input/SharedTask/thunder/Bitext_v1_sup/amen/train.decup.am-en.en", help="input files")
    args = parser.parse_args()

    seen = set()
    with open(args.src, "r", encoding="utf-8") as src_r:
        with open(args.tgt, "r", encoding="utf-8") as tgt_r:
            pool = Pool(args.workers)
            src_lines = src_r.readlines()
            tgt_lines = tgt_r.readlines()
            results = list(pool.imap(get_hashes_and_lines, src_lines, 1000))
            rm_count = 0
            with open(args.new_src, "w", encoding="utf-8") as src_w:
                with open(args.new_tgt, "w", encoding="utf-8") as tgt_w:
                    for i, hash in enumerate(results):
                        if hash not in seen:
                            seen.add(hash)
                            src_w.write(src_lines[i])
                            tgt_w.write(tgt_lines[i])
                        else:
                            #print("{} | {} | {}".format(i, hash, src_lines[i]))
                            rm_count += 1
                        if i % 5000000 == 0:
                            print("Processing {} lines | removing duplicated pairs {}".format(i, rm_count))
            print("Remaining {} lines".format(len(src_lines) - rm_count))
            print("{} -> {}".format(args.src, args.new_src))
            print("{} -> {}".format(args.tgt, args.new_tgt))


if __name__ == "__main__":
    main()