import sys
import argparse
import karnak.util.athena as ka
import karnak.util.log as klog

def run(args):
    parser = argparse.ArgumentParser(description='karnak - aws athena query test')
    parser.add_argument('sql', type=str)
    parser.add_argument('--database', type=str)
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--workgroup', type=str, default='primary')
    parser.add_argument('--method', type=str, choices=['rest', 'jdbc', 'csv'], default='rest')
    parser.add_argument('--verbosity', '-v', type=int, default=1,
                        help='verbosity level 0-5, default 1, larger means more verbose')
    args = parser.parse_args(args)
    klog.set_log_level(args.verbosity)

    df = ka.select_pd(sql=args.sql, aws_region=args.region, database=args.database,
                      workgroup=args.workgroup, method=args.method)
    print(df)

if __name__ == "__main__":
    run(sys.argv[1:])
