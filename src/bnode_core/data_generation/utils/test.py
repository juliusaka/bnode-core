import sys

if __name__ == '__main__':
    print('Test sys.argv:')
    print(sys.argv)
    # try to remove argument '--cfg'
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--cfg'):
            print(f'Removing argument: {arg}')
            sys.argv.pop(i)
            break
    print('Modified sys.argv:')
    print(sys.argv)