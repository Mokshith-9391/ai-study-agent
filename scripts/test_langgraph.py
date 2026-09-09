from src.graph.workflow import build_study_graph


def main() -> None:
    graph = build_study_graph()

    print("GRAPH COMPILED")

    print("\nGRAPH NODES:")
    print(
        list(
            graph.get_graph().nodes.keys()
        )
    )

    print("\nGRAPH EDGES:")
    for edge in graph.get_graph().edges:
        print(
            f"{edge.source} -> {edge.target}"
        )

    print("\nLANGGRAPH FOUNDATION TEST PASSED")


if __name__ == "__main__":
    main()