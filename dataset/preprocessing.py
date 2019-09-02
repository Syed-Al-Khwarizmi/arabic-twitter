import os


def main():
	path=os.getcwd()+"\\Negative\\negative1.txt"	
	f = open(path,'r',encoding='utf8')
	line=f.read()
	print(line)

if __name__ == "__main__":
	main();
