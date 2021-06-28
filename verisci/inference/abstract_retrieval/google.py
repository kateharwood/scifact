import json
import argparse

from google_utils import getDocsBatch, GoogleConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--cse_id', type=str)
    #parser.add_argument('--cuda_device', type=int, default=-1)

    args = parser.parse_args()

    # with open(args.config) as f:
    #     config = json.load(f)
    
    google_config = GoogleConfig(args.api_key, args.cse_id)
    
    with open(args.out_file, 'w') as outfile:
        for docs in getDocsBatch(args.in_file, google_config):
            print(json.dumps(docs), file=outfile)
                            
