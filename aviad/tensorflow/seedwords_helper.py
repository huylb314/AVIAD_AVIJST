def string_to_list(a_string):
    return a_string.replace('\n','').split(',')

def read_file_to_list(a_file_location):
    file_reader = open(a_file_location,"r")
    the_return_list = []
    return [string_to_list(the_line) for the_line in file_reader]
