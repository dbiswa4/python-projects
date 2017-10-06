class SimpleFactory(object):

    @staticmethod  # This decorator allows to run method without class instance, .e. SimpleFactory.build_connection
    def build_connection(protocol):
        if protocol == 'http':
            return HTTPConnection()
        elif protocol == 'ftp':
            return FTPConnection()