import json
import sys

#python json_args.py '{"feed_name":"adcom-imp","conf_path":"s3://aolp-dev-lsa-cls-meta-qa/conf"}'

if __name__ == "__main__":
    print sys.argv[1]
    data=json.loads(sys.argv[1])
    print data
    print data['feed_name']
    print data['conf_path']