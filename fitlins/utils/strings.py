import re


def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def to_alphanum(string):
    """Convert string to alphanumeric

    Replaces all other characters with underscores and then converts to camelCase

    Examples
    --------

    >>> to_alphanum('abc123')
    'abc123'
    >>> to_alphanum('a four word phrase')
    'aFourWordPhrase'
    >>> to_alphanum('hyphen-separated')
    'hyphenSeparated'
    >>> to_alphanum('object.attribute')
    'objectAttribute'
    >>> to_alphanum('array[index]')
    'arrayIndex'
    >>> to_alphanum('array[0]')
    'array0'
    >>> to_alphanum('snake_case')
    'snakeCase'
    """
    return snake_to_camel(re.sub("[^a-zA-Z0-9]", "_", string))
