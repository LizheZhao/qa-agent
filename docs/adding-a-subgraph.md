# Adding a subgraph

A subgraph is a private nested workflow owned and versioned by its parent agent. Put it in the parent
package's `subgraphs/` directory, give it private typed state where useful, compose it from the parent
graph, and test both its isolated transitions and parent integration. It receives dependencies
explicitly and does not get a deployment-manifest entry or independent version.

If the component needs its own public request/result contract, release lifecycle, tool permissions, or
trace identity, it is a subagent instead.

