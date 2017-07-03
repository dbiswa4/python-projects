import boto3
import os
import time
from time import mktime
#from datetime import datetime
import datetime


def get_s3_total_files_size(bucket, prefix):
    total_size = 0
    # Size of all objects with a prefix in a bucket
    for obj in boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix):
        total_size += obj.size

    return total_size


def get_s3_last_modified_time(bucket, prefix):
    maxtime = '0001-01-01 01:01:01'

    max_datetime = time.strptime(maxtime, "%Y-%m-%d %H:%M:%S")

    print 'max_datetime initializd to : ', str(max_datetime)

    for obj in boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix):
        last_modified_time = obj.last_modified
        print "time : ", last_modified_time
        print 'type of last_modified_time : ', type(last_modified_time)
        print "obj : ", obj
        print "bucket : ", obj.bucket_name
        print "key : ", obj.key

        # modified = time.strptime(obj.last_modified, '%a, %d %b %Y %H:%M:%S %Z')
        # print 'modified : ', modified
        # print 'strptime : ', time.strptime(obj.last_modified[:19], "%Y-%m-%dT%H:%M:%S")
        # convert to datetime
        # dt = datetime.fromtimestamp(mktime(modified))
        # print 'dt : ', dt

        print 'last_modified_time[:19] : ', str(last_modified_time)[:19]

        naive_last_modified_time = time.strptime(str(last_modified_time)[:19], "%Y-%m-%d %H:%M:%S")

        print 'naive_last_modified_time : ', str(naive_last_modified_time)

        if naive_last_modified_time > max_datetime:
            max_datetime = naive_last_modified_time

    print '\nmaxtime : ', str(max_datetime)

def get_s3_last_modified_time_other(bucket, prefix):
    print '\n\nIn get_s3_last_modified_time_other()...'
    #maxtime = '0001-01-01 01:01:01'
    #max_datetime = time.strptime(maxtime, "%Y-%m-%d %H:%M:%S")
    maxtime = datetime.MINYEAR
    max_datetime = datetime.MINYEAR

    maxtime_key = 'none'

    print 'max_datetime initializd to : ', str(max_datetime)

    for obj in boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix):
        last_modified_time = obj.last_modified
        print "time : ", last_modified_time
        print 'type of last_modified_time : ', type(last_modified_time)
        print "obj : ", obj
        print "bucket : ", obj.bucket_name
        print "key : ", obj.key

        # modified = time.strptime(obj.last_modified, '%a, %d %b %Y %H:%M:%S %Z')
        # print 'modified : ', modified
        # print 'strptime : ', time.strptime(obj.last_modified[:19], "%Y-%m-%dT%H:%M:%S")
        # convert to datetime
        # dt = datetime.fromtimestamp(mktime(modified))
        # print 'dt : ', dt

        print 'last_modified_time[:19] : ', str(last_modified_time)[:19]

        naive_last_modified_time = time.strptime(str(last_modified_time)[:19], "%Y-%m-%d %H:%M:%S")

        print 'naive_last_modified_time : ', str(naive_last_modified_time)

        if naive_last_modified_time > max_datetime:
            max_datetime = naive_last_modified_time
            maxtime_key = obj.key


    print '\n\nmaxtime : ', str(max_datetime)
    print 'maxtime_key : ', maxtime_key


def get_s3_objs(bucket, prefix):
    for obj in boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix):
        print 'key : ', obj.key

def get_s3_sub_folders_other(bucket, prefix):
    #Does not give the result expected
    print '\n\n***in get_s3_sub_folders() ***'
    for obj in boto3.resource('s3').Bucket(bucket).objects.filter(Delimiter='/'):
        print 'key : ', obj.key


def get_s3_sub_folders(bucket, prefix):
    print '\n\n***in get_s3_sub_folders_new2() ***'
    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    for o in result.get('CommonPrefixes'):
        print 'sub folder : ', o.get('Prefix')


if __name__ == '__main__':
    _SCRIPT_NAME = os.path.basename(__file__)
    print '_SCRIPT_NAME : ', _SCRIPT_NAME

    # Bucket/Folder size test
    size = get_s3_total_files_size("techknowera-dev-bucket", "profiles/2016-08-16")
    print 'size : ', str(size)

    size = get_s3_total_files_size("techknowera-dev-bucket", "profiles/2016-08-16/")
    print 'size : ', str(size)

    size = get_s3_total_files_size("techknowera-dev-bucket", "profiles/2016-08-16/2016-08-17_00-20-20")
    print 'size : ', str(size)

    # Last Modified Time Tests
    get_s3_last_modified_time("techknowera-dev-bucket-std", "run")

    get_s3_last_modified_time_other("techknowera-dev-bucket-std", "run")

    # Get s3 sub folders
    get_s3_sub_folders_other("techknowera-dev-bucket-std", "run")

    get_s3_sub_folders("techknowera-dev-bucket-std", "run/")

