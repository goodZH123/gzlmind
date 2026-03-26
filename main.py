import torch    
def main():
    print("Hello, gzlMind!")
    seq_len = 5
    x = torch.triu(
                torch.full((seq_len, seq_len), float("-inf")),
                diagonal=1,
            )
    print(x)
if __name__ == "__main__":
    main()
