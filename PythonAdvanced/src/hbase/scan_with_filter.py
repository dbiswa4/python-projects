import happybase


group_table = "meetup_groups"
localhost = '108.161.128.86'
connection = happybase.Connection(localhost)
group_con = connection.table(group_table)

#g = group_con.scan(filter="SingleColumnValueFilter ('group_details', 'city', =, 'substring:phoenix')")
#for key, data in g:
#    print key

city = 'new york'
query = "SingleColumnValueFilter ('group_details', 'city', =, 'substring:" + city + "')"
#Below does not work
#query = "SingleColumnValueFilter ('group_details', 'city', =, 'binary:" + city + "')"

print 'query : ', query

g = group_con.scan(filter=query)
for key, data in g:
    print key
    print 'data type of key : ', type(key)

