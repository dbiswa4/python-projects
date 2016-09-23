import re

def regex_match():
    print '\nregex_match()...'
    line = "Cats are smarter than dogs"
    #re.M -> multiline
    #re.I -> ignorecase
    matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)

    if matchObj:
        print 'matchObj.group() : ', matchObj.group()
        print 'matchObj.group(1) : ', matchObj.group(1)
        print 'matchObj.group(2) : ', matchObj.group(2)
        #Interesting - it retuns 26, which is the length of the whole line
        #I was expexting it to return 5
        print 'Length of matchObj.group() : ', len(matchObj.group())
    else :
        print 'No Match...'

def regex_search():
    print '\nregex_search()...'
    line = "Cats are smarter than dogs"
    searchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)

    if searchObj:
        print 'searchObj.group()  : ', searchObj.group()
        print 'searchObj.group(1) : ', searchObj.group(1)
        print 'searchObj.group(2) : ', searchObj.group(2)
    else:
        print 'Nothing found...'

def match_vs_search():
    '''
    match checks for a match only at the beginning of the string, while search checks for a match anywhere in the
    string (this is what Perl does by default).
    '''
    print '\nmatch_vs_search()...'
    line = "Cats are smarter than dogs"
    matchObj = re.match(r'dogs', line, re.M | re.I)
    if matchObj:
        print "match --> matchObj.group() : ", matchObj.group()
    else:
        print "No match!!"

    searchObj = re.search(r'dogs', line, re.M | re.I)
    if searchObj:
        print "search --> searchObj.group() : ", searchObj.group()
    else:
        print "Nothing found!!"


if __name__ == '__main__':
    print 'Regex Match example'
    regex_match()
    regex_search()
    match_vs_search()