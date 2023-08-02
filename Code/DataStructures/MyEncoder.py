from _ctypes import PyObj_FromPtr
import json
import re

"""
code origin from martineau of https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module  
copied at the 02.06.2023
"""

class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr


if __name__ == '__main__':
    from string import ascii_lowercase as letters

    data_structure = {
        'layer1': {
            'layer2': {
                'layer3_1': NoIndent([{"x":1,"y":7}, {"x":0,"y":4}, {"x":5,"y":3},
                                      {"x":6,"y":9},
                                      {k: v for v, k in enumerate(letters)}]),
                'layer3_2': 'string',
                'layer3_3': NoIndent([{"x":2,"y":8,"z":3}, {"x":1,"y":5,"z":4},
                                      {"x":6,"y":9,"z":8}]),
                'layer3_4': NoIndent(list(range(20))),
            }
        }
    }

    print(json.dumps(data_structure, cls=MyEncoder, sort_keys=True, indent=2))
