"""One command: clean -> features -> baseline vs model -> ship gate."""
from brain import clean, train

if __name__ == "__main__":
    hourly = clean.main()
    train.main(hourly)
