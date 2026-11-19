"""One command: clean -> features -> baseline vs model -> ship gate -> plot."""
from brain import clean, plot, train

if __name__ == "__main__":
    hourly = clean.main()
    train.main(hourly)
    plot.main()
