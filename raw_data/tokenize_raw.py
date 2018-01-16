from nltk.tokenize.moses import MosesTokenizer
import argparse

parser = argparse.ArgumentParser("tokenize_raw.py")
parser.add_argument('--file_paths', type=str, nargs='+',
                    help='train/dev/test file paths')
parser.add_argument('--lowercase', action='store_false',
                    help='lowercase text in default')

args = parser.parse_args()
# print(type(args.file_paths))
# print(args.file_paths)

if __name__ == '__main__':

    t = MosesTokenizer()

    for file_path in args.file_paths:
        with open(file_path, 'r') as f:

            tokenized_lines = []

            lines = f.readlines()
            for line in lines:
                line_split = line.split('|||')
                if len(line_split) != 3:
                    print(line_split)
                    raise Exception('Each line should contain 3 arguments')
                arg1_tokens = t.tokenize(line_split[1])
                arg2_tokens = t.tokenize(line_split[2])
                arg1 = ' '.join(arg1_tokens)
                arg2 = ' '.join(arg2_tokens)
                new_line = line_split[0] + '|||' + arg1 + '|||' + arg2 + '\n'
                if args.lowercase:
                    tokenized_lines.append(new_line.lower())
                else:
                    tokenized_lines.append(new_line)

            with open(file_path + '.tok', 'w') as f_new:
                f_new.writelines(tokenized_lines)