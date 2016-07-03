from datetime import datetime, timedelta

def get_prev_date(go_back_days):
    print datetime.utcnow().strftime("%Y-%m-%d")
    prev_date =  datetime.utcnow() - timedelta(days=go_back_days)
    print prev_date
    print prev_date.strftime("%Y-%m-%d")
    return prev_date.strftime("%Y-%m-%d")

def get_prev_date_again(go_back_days, date_fmt_str="%Y-%m-%d"):
    print datetime.utcnow().strftime(date_fmt_str)
    prev_date =  datetime.utcnow() - timedelta(days=go_back_days)
    print prev_date
    print prev_date.strftime(date_fmt_str)
    return prev_date.strftime(date_fmt_str)

if __name__ == "__main__":
    print "datetime module"
    #get_prev_date()
    prev_date = get_prev_date(7)
    print "prev_date : ", prev_date
    
    prev_date = get_prev_date_again(7, "%Y-%m-%d")
    print "prev_date : ", prev_date
    
    prev_date = get_prev_date_again(7)
    print "prev_date : ", prev_date