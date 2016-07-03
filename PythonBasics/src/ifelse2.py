def else_if(shard_size_gb = 0):
    
    if shard_size_gb < 20:
        core_count = 10
    elif (shard_size_gb > 20 and shard_size_gb < 200):
        core_count = 20
    elif (shard_size_gb > 200 and shard_size_gb < 600):
        core_count = 30
    elif (shard_size_gb > 600 and shard_size_gb < 1024):
        core_count = 50
    elif (shard_size_gb > 1024 and shard_size_gb < 2048):
        core_count = 70
    else:
        core_count = 90
    
    return core_count
    
    print "core_count : ", core_count
            
if __name__ == "__main__":
    print "Hello World"

    core_count=else_if(700)
    print "core_count : ", core_count