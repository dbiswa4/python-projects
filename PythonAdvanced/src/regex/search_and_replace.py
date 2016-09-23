import re

def replace_demo():
    print '\nrepalce_demo()...'
    phone = "2004-959-559 # This is Phone Number"

    # Delete Python-style comments
    num = re.sub(r'#.*$', "", phone)
    print 'Phone Num : ', num

    # Remove anything other than digits
    num = re.sub(r'\D', "", phone)
    print "\nPhone Num : ", num


if __name__ == '__main__':
    print 'Search and Replace demo...'
    replace_demo()