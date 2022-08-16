from collections import OrderedDict

from mysql.connector import MySQLConnection, Error


class Mysql:

    __instance   = None
    __host       = None
    __user       = None
    __password   = None
    __database   = None
    __session    = None
    __connection = None
    __dictionary = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance or not cls.__database:
            cls.__instance = super(Mysql, cls).__new__(cls)  # , *args, **kwargs)
        return cls.__instance

    def __init__(self, host='localhost', user='root', password='', database=''):
        self.__host     = host
        self.__user     = user
        self.__password = password
        self.__database = database

    def __open(self):
        cnx = None
        try:
            cnx = MySQLConnection(host=self.__host, user=self.__user, password=self.__password, database=self.__database)
            self.__connection = cnx
            self.__session    = cnx.cursor(dictionary=self.__dictionary)
        except Error as e:
            print('Error %d: %s' % (e.args[0], e.args[1]))

    def __close(self):
        self.__session.close()
        self.__connection.close()
        
    def dictionary_on(self):
        self.__dictionary = True
        
    def dictionary_off(self):
        self.__dictionary = False
        
    def concatenate_table_columns(self, source, columns):
        table_columns = []
        for column in columns:
            table_columns.append(source + '.' + column)
        return tuple(table_columns)

    def select(self, table, where=None, *args, **kwargs):
        result = None
        query = 'SELECT '
        if not args:
            print('TRUE')
            keys = ['*']
        else:
            keys = args
        print(keys)
        values = tuple(kwargs.values())
        l = len(keys) - 1

        for i, key in enumerate(keys):
            query += key
            if i < l:
                query += ","

        query += ' FROM %s' % table

        if where:
            query += " WHERE %s" % where

        print(query)
        self.__open()
        self.__session.execute(query, values)
#         number_rows = self.__session.rowcount
#         number_columns = len(self.__session.description)
#         print(number_rows, number_columns)
#         if number_rows >= 1 and number_columns > 1:
        result = [item for item in self.__session.fetchall()]
#         else:
#             print('if only')
#             result = [item[0] for item in self.__session.fetchall()]
        self.__close()

        return result
    
    def join(self, table1, table2, where=None, join_type='inner'):
        """SELECT
            m.member_id, 
            m.name member, 
            c.committee_id, 
            c.name committee
        FROM
            members m
        INNER JOIN committees c USING(name);"""

    def insert_into(self, table, columns, source=None, where=None, equivalents=None, *args, **kwargs):           
        query = "INSERT INTO %s SET " % table
        table_columns = self.concatenate_table_columns(table, columns)
        
        if source:
            column_equivalents = self.concatenate_table_columns(source, columns)
            # column_and_equivalents = [(columns[i], column_equivalents[i]) for i in range(len(columns))]
            pre_query_columns = [column + ' = %s' for column in table_columns]
            pre_query_columns = [pre_query_columns[i] % column_equivalents[i] for i in range(len(column_equivalents))]
            # print(pre_query_columns)
            query += ", ".join(pre_query_columns) # + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
        
        if kwargs:
            keys   = kwargs.keys()
            values = tuple(kwargs.values()) + tuple(args)
            l = len(keys) - 1
            for i, key in enumerate(keys):
                query += key+ " = %s"
                if i < l:
                    query += ","
                ## End if i less than 1
            ## End for keys
        if where:
            query += " WHERE %s" % where
            
#         print(query)
#         stop = columns[30]
        self.__open()
        self.__session.execute(query) #, values)
        self.__connection.commit()

        # Obtain rows affected
        update_rows = self.__session.rowcount
        self.__close()

        return update_rows

    def insert(self, table, columns, values, *args, **kwargs):
        # values = None
        query = "INSERT INTO %s " % table
        query += "(" + ",".join(["%s"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"

        if kwargs:
            keys = kwargs.keys()
            values = tuple(kwargs.values())
            query += "(" + ",".join(["%s"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
        elif args:
            values = args
            query += " VALUES(" + ",".join(["%s"] * len(values[0][0])) + ")"
        
#         print(query)
#         print(values)

        self.__open()
        self.__session.execute(query, values)
        self.__connection.commit()
        self.__close()
        
        return self.__session.lastrowid
    
    def insert_multiple(self, table, columns, values, *args, **kwargs):
        # values = None
        query = "INSERT INTO %s " % table
        query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")"
        
        if kwargs:
            keys = kwargs.keys()
            values = tuple(kwargs.values())
            query += "(" + ",".join(["`%s`"] * len(keys)) % tuple(keys) + ") VALUES (" + ",".join(["%s"] * len(values)) + ")"
        elif args:
            query += "(" + ",".join(["`%s`"] * len(columns)) % tuple(columns) + ")"
            query += " VALUES(" + ",".join(["%s"] * len(args)) + ")"
        
        # print(query)
        # print(values[0][0])
        self.__open()
        self.__session.executemany(query, values)
        self.__connection.commit()
        self.__close()
        
        return self.__session.lastrowid

    def delete(self, table, where=None, *args):
        query = "DELETE FROM %s" % table
        if where:
            query += ' WHERE %s' % where

        values = tuple(args)

        self.__open()
        self.__session.execute(query, values)
        self.__connection.commit()

        # Obtain rows affected
        delete_rows = self.__session.rowcount
        self.__close()

        return delete_rows

    def select_advanced(self, sql, *args):
        od = OrderedDict(args)
        query  = sql
        values = tuple(od.values())
        self.__open()
        self.__session.execute(query, values)
        number_rows = self.__session.rowcount
        number_columns = len(self.__session.description)

        if number_rows >= 1 and number_columns > 1:
            result = [item for item in self.__session.fetchall()]
        else:
            result = [item[0] for item in self.__session.fetchall()]

        self.__close()
        
        return result
