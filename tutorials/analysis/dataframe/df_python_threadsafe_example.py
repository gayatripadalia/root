## \file
## \ingroup tutorial_dataframe
## \notebook -nodraw
##
## Demonstrates a thread-safe column definition in Python using `rdfslot_`.
##
## In C++, ROOT provides `DefineSlot` and `RedefineSlot` for thread-safe operations.
## However, these APIs are not currently available in Python.
##
## This example shows how to achieve similar behavior using `rdfslot_`,
## which represents the slot (thread) ID assigned during execution.
##
## This approach ensures that each thread works with its own data,
## avoiding race conditions in multi-threaded environments.
import ROOT
def main():
    # Enable implicit multi-threading
    ROOT.ROOT.EnableImplicitMT()
    print("Multithreading Enabled:", ROOT.ROOT.IsImplicitMTEnabled())
    # Create a simple RDataFrame with 20 entries
    df = ROOT.RDataFrame(20)
    # Define thread-safe columns using rdfslot_
    df = (
        df.Define("thread_id", "rdfslot_")
          .Define("value", "rdfslot_ * 10")
          .Define("combined", "thread_id + value")
    )
    print("\nDisplaying DataFrame with thread-aware columns:\n")
    # Display results
    df.Display(["thread_id", "value", "combined"]).Print()
    print("\nExplanation:")
    print("- 'thread_id' shows which thread processed each entry.")
    print("- 'value' is derived independently per thread.")
    print("- 'combined' demonstrates safe computation using thread-local values.")
if __name__ == "__main__":
    main()
