import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str,
                        default=r'/home/v-jiaya/SharedTask/log/log.txt', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.log, "r", encoding="utf-8") as r:
        lines = r.readlines()
        x2x = []
        for line in lines:
            if 'BLEU+case.mixed' in line:
                score = float(line.split()[4])
                x2x.append(score)