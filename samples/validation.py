def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        
        if len(lines1) != len(lines2):
            print("Files have different number of lines.")
            return False
        array_q=[]
        array_i=[]
        for line_num, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
            numberl_quafu=line1.split()
            numberl_ingress=line2.split()
            num_quafu=[int(num) for num in numberl_quafu]
            #array_q.append(num_quafu)
            num_ingress=[int(num) for num in numberl_ingress]
            #array_i.append(num_quafu)
            if line1 != line2:
                print(f"Line {line_num} differs:")
                print(f"quafu 1: {line1.strip()}")
                print(f"ingress 2: {line2.strip()}")
                print((num_quafu[0]+num_quafu[2])%128+1,(num_ingress[0]+num_ingress[2])%128+1)
                print()
        
        print("Files are identical.")
        return True

def main():
    file1 = "/home/hdd/yongze/datasets/quafu-3.txt"
    file2 = "/home/hdd/yongze/datasets/ingress-static.txt"
    
    if compare_files(file1, file2):
        print("The content of the files is the same.")
    else:
        print("The content of the files is different.")

if __name__ == "__main__":
    main()
