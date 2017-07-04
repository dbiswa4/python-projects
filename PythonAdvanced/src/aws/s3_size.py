import boto3
import os
import time
from time import mktime
from datetime import datetime
#import datetime


def get_s3_total_files_size(bucket, prefix):
    total_size = 0
    # Size of all objects with a prefix in a bucket
    for obj in boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix):
        total_size += obj.size

    return total_size

if __name__ == '__main__':

    bucket = "my-bucket-name"
    prefix1 = "file-prefix"
    date = "2017/06/29"
    postfix = "file-postfix"

    print 'print sizes : '
    print 'current utc time : ', datetime.utcnow()

    sizes = []

    for i in range(00, 24):
        #print '%02d' % i
        hp = '{:02d}'.format(i)
        #print 'hp : ', hp
        prefix = prefix1 + "/" + date + "/" + hp + "/" + postfix + "/"
        #print prefix
        size = round(get_s3_total_files_size(bucket, prefix) / float(1024*1024*1024), 2)
        #print 'size : ', str(size)

        folder = "s3://" + bucket + "/" + prefix

        print folder + "," + str(size)

        sizes.append(size)

    print sizes

    for s in sizes:
        print str(s)
