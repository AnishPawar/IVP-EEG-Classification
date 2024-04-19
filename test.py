import argparse

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Add two numbers")

    # Add arguments
    parser.add_argument("number1", type=float, help="First number")
    parser.add_argument("number2", type=float, help="Second number")

    # Parse arguments
    args = parser.parse_args()

    # Calculate the sum
    result = args.number1 + args.number2

    # Print the result
    print(f"The sum of {args.number1} and {args.number2} is {result}")

if __name__ == "__main__":
    main()
