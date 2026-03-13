import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_solution(
    solution: dict,
    instance,
    title: str = "VRP Solution",
    output_path: str = "solution.png",
) -> None:
    customers = instance.customers
    routes = solution["routes"]
    colours = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    for r_idx, route in enumerate(routes):
        colour = colours[r_idx % len(colours)]
        xs = [customers[n].x_coordinate for n in route]
        ys = [customers[n].y_coordinate for n in route]
        ax.plot(xs, ys, "-o", color=colour, linewidth=1.5, markersize=5, zorder=2)

    depot = customers[0]
    ax.plot(
        depot.x_coordinate, depot.y_coordinate,
        "*", color="black", markersize=14, zorder=4, label="Depot",
    )

    for c in customers[1:]:
        ax.plot(c.x_coordinate, c.y_coordinate, "o", color="steelblue", markersize=8, zorder=3)
        ax.annotate(
            str(c.customer_id),
            (c.x_coordinate, c.y_coordinate),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            zorder=5,
        )

    patches = [
        mpatches.Patch(color=colours[i % len(colours)], label=f"Route {i + 1}")
        for i in range(len(routes))
    ]
    patches.insert(0, mpatches.Patch(color="black", label="Depot"))

    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_title(f"{title}\nTotal distance: {solution['total_distance']:.4f}", fontsize=12)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)