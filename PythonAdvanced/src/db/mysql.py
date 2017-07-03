import MySQLdb

def get_db_connection(db_config):
    try:

        db_connection = MySQLdb.connect(**db_config)

    except MySQLdb.Error as err:

        print("MySQLdb.Error: " + err.message)
        raise err  # Re-raise the exception

    print("Created database connection : " + str(db_connection))
    return db_connection

def get_table_data(db_config):
    query_string = "select cust_id,cust_name from customer"
    print("query_string : " + query_string)
    con = get_db_connection(db_config)
    cursor = con.cursor()
    cursor.execute(query_string)
    # Fetch all the rows
    results = cursor.fetchall()

    file_name = "./table-data.csv"
    fh = open(file_name, 'w')

    for row in results:
        seg = row[0]
        elem = row[1]
        rec = str(seg) + "," + str(elem) + ","
        fh.write("%s\n" % rec)

    fh.close()


if __name__ == '__main__':
    print 'MYSQL download'
    db_config = {"host": "mysql.techknowera.com", "port": 3306, "user": "username", "passwd": "passwd","db": "dbname"}
    get_table_data(db_config)

