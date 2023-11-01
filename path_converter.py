#convinience script to convert Windows paths to Unix paths and vice versa

import os
import sys

def map_network_paths(path):
    """
    Map network paths: \\production.dnae.emea.denso to /mnt/networks and vice versa.
    """
    if path.startswith('/mnt/networks'):
        path = path.replace('/mnt/networks', '\\\\production.dnae.emea.denso')
    elif path.startswith('\\production.dnae.emea.denso'):
        path = path.replace('\\production.dnae.emea.denso', '/mnt/networks')
    elif path.startswith('\\\\production.dnae.emea.denso'):
        path = path.replace('\\\\production.dnae.emea.denso', '/mnt/networks')
    return path

def windows_to_unix_path(path):
    """
    Convert a Windows path to a Unix path.
    """
    # Map network paths
    path = map_network_paths(path)

    # Replace backslashes with forward slashes
    path = path.replace('\\', '/')
    
    # Remove leading double slashes, if any
    if path.startswith('//'):
        path = '/' + path.lstrip('/')

    return path

def unix_to_windows_path(path):
    """
    Convert a Unix path to a Windows path.
    """
    # Map network paths
    path = map_network_paths(path)
    
    # Replace forward slashes with backslashes
    path = path.replace('/', '\\')

    return path

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("To convert Windows path to Unix path: path_converter.py w2l <WindowsPath>")
        print("To convert Unix path to Windows path: path_converter.py l2w <UnixPath>")
        sys.exit(1)

    action = sys.argv[1]
    path = sys.argv[2]

    # print('input path: ', path)

    if action == "w2l":
        unix_path = windows_to_unix_path(path)
        print(unix_path)
    elif action == "l2w":
        windows_path = unix_to_windows_path(path)
        print(windows_path)
    else:
        print("Invalid action. Use 'w2l' for Windows to Unix conversion or 'l2w' for Unix to Windows conversion.")

if __name__ == "__main__":
    main()
