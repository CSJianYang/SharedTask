import argparse
import sentencepiece
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/large-scale/flores101_dataset/devtest/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/large-scale/flores101_dataset/devtest-code/', help='input stream')
    args = parser.parse_args()
    return args