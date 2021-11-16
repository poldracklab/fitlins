def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def to_alphanum(string):
    string.replace('.', '_')
    string.replace('-', '_')
    return snake_to_camel(string)
