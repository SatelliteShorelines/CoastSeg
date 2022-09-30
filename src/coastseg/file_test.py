import os 

# os.path.abspath(os.path.join(os.getcwd(),'logs'))
print( os.path.abspath(os.getcwd()))
print( os.path.abspath('.'))
print(os.path.abspath(os.path.join(os.getcwd(),'src','coastseg')))
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
print( script_dir)
rel_path = "transects"
abs_file_path = os.path.join(script_dir, rel_path)
print( abs_file_path)
file_dir = os.path.dirname(os.path.abspath(__file__))
print (file_dir)