# this file is used to convert the input data from jsonl to json format

import json
import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python to_json.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # check that input file is a jsonl file
    if not input_file.endswith('.jsonl'):
        print("Input file must be a jsonl file")
        sys.exit(1)
    
    # check that output file is a json file
    if not output_file.endswith('.json'):
        print("Output file must be a json file")
        sys.exit(1)
    
    # check that input file exists
    if not os.path.exists(input_file):
        print("Input file does not exist")
        sys.exit(1)
    
    # check that output file does not exist
    if os.path.exists(output_file):
        print("Output file already exists")
        sys.exit(1)

    print("Converting {} to {}".format(input_file, output_file))

    with open(input_file, 'r') as f:
        with open(output_file, 'w') as out:
            for line in f:
                json.dump(json.loads(line), out)
                out.write('\n')
    
    if os.path.exists(output_file):
        print("Conversion successful")

if __name__ == "__main__":
    main()