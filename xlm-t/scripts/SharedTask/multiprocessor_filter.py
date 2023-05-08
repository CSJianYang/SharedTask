import multiprocessing as MP
from multiprocessing import Pool
import math
import argparse
import os
import re
import tqdm
import traceback
import linecache
import langid
import requests
import nltk
import math
import json
import re
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download('stopwords')

language2simply = {"chinese": "zh", "english": "en"}
class MPLogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def error(msg, *args):
        return MP.get_logger().error(msg, *args)

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result


def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()

class MultiProcessor(object):
    def __init__(self, args):
        self.workers = args.workers
        self.length_ratio = args.length_ratio
        self.workers = args.workers


    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--src', '-src', type=str,
                            default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/sr/WikiMatrix.et-sr.et.filt', help='input stream')
        parser.add_argument('--tgt', '-tgt', type=str,
                            default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/sr/WikiMatrix.en-sr.sr.filt', help='input stream')
        parser.add_argument('--new-src', '-new-src', type=str,
                            default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/train-filter/test.en', help='input stream')
        parser.add_argument('--new-tgt', '-new-tgt', type=str,
                            default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/train-filter/test.sr', help='input stream')
        parser.add_argument('--workers', '-workers', type=int, default=40, help='')
        parser.add_argument('--length-ratio', '-length-ratio', type=float, default=2.0, help='')
        args = parser.parse_args()
        print(args)
        return args



    def single_worker(
            self, src_lines, tgt_lines, worker_id=0, src_lang="en", tgt_lang="sr"
    ):
        rm_count = 0
        results = []
        for count, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
            detok_src = decode(src_line)
            detok_tgt = decode(tgt_line)
            if count % 1000 == 0 and worker_id == 0:
                #print(u"Complete processing {} examples | removing {} examples".format(count, rm_count), end="\r")
                print(u"Complete processing {} examples | removing {} examples".format(count, rm_count))
            count += 1
            if len(detok_tgt.split()) == 0 or len(detok_src.split()) == 0:
                rm_count += 1
                continue
            if len(detok_src.split()) / len(detok_tgt.split()) > self.length_ratio or len(detok_tgt.split()) / len(detok_src.split()) > self.length_ratio:
                rm_count += 1
                continue
            infer_lang = langid.classify(detok_tgt)[0]
            if infer_lang != tgt_lang:  # langid.classify(detok_src)[0] != src_lang or
                rm_count += 1
                continue
            results.append((src_line, tgt_line))
        return results


    def process_file(self, src, tgt, new_src, new_tgt, src_lang, tgt_lang):
        with open(src, "r", encoding="utf-8") as src_r:
            with open(tgt, "r", encoding="utf-8") as tgt_r:
                src_lines = src_r.readlines()
                tgt_lines = tgt_r.readlines()
                block_lines = math.ceil(len(src_lines)/self.workers)
                p = Pool(self.workers)
                results = []
                for worker_id in range(self.workers):
                    sub_src_lines = src_lines[block_lines * worker_id: block_lines * (worker_id + 1)]
                    sub_tgt_lines = tgt_lines[block_lines * worker_id: block_lines * (worker_id + 1)]
                    results.append(p.apply_async(MPLogExceptions(self.single_worker), args=(sub_src_lines, sub_tgt_lines, worker_id, src_lang, tgt_lang)))
                p.close()
                p.join()
                all_results = []
                for result in results:
                    all_results.extend(result.get())
                src_lines = [r[0] for r in all_results]
                tgt_lines = [r[1] for r in all_results]
                assert len(results) == self.workers, "Please check the number of results from subprocess !"
                if not os.path.exists(os.path.dirname(new_src)):
                    os.makedirs(os.path.dirname(new_src))
                with open(new_src, "w", encoding="utf-8") as src_w:
                    with open(new_tgt, "w", encoding="utf-8") as tgt_w:
                        src_w.write("".join(src_lines))
                        tgt_w.write("".join(tgt_lines))
                        print("Successfully writing to {} and {}".format(new_src, new_tgt))
                print("{} | All Subprocess Context files Processes Finished !".format(len(src_lines)))


    def process_files(self, file_list):
        for src_set, new_src_set, vocab, dict_path, src_language, tgt_language in file_list:
            self.process_file(src_set, new_src_set, vocab, dict_path, src_language, tgt_language)


if __name__ == "__main__":
    args = MultiProcessor.parse_args()
    mp = MultiProcessor(args)
    mp.process_file(args.src, args.tgt, args.new_src, args.new_tgt, args.src.split('.')[-1], args.tgt.split('.')[-1])



















