import parser

parser.parse('train.png', mode = 'train')
result = parser.parse('code.png')

print(result)