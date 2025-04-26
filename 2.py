import math
import random
import sys
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
                             QTextEdit, QTableWidget, QTableWidgetItem, QGraphicsScene, QGraphicsView, QCheckBox,
                             QLineEdit)
from PyQt5.QtGui import QPen, QBrush, QPainter, QPolygonF
from PyQt5.QtCore import Qt, QPointF


class TravelingSalesmanApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        self.setWindowTitle("Traveling Salesman Problem")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        mainLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()
        rightLayout = QVBoxLayout()

        self.calculateRouteButton = QPushButton("Calculate Route")
        self.calculateRouteButton.clicked.connect(self.calculateOptimalRoute)

        self.undoLastActionButton = QPushButton("Undo")
        self.undoLastActionButton.clicked.connect(self.undoLastAction)

        self.clearGraphButton = QPushButton("Clear Graph")
        self.clearGraphButton.clicked.connect(self.clearGraph)

        self.useModificationCheckBox = QCheckBox("Use Modification")

        self.resultDisplay = QTextEdit()
        self.resultDisplay.setReadOnly(True)

        self.graphDisplayScene = QGraphicsScene()
        self.graphDisplayView = QGraphicsView(self.graphDisplayScene)
        self.graphDisplayView.setRenderHint(QPainter.Antialiasing)
        self.graphDisplayView.setMouseTracking(True)
        self.graphDisplayView.mousePressEvent = self.handleGraphClick

        self.solutionDisplayScene = QGraphicsScene()
        self.solutionDisplayView = QGraphicsView(self.solutionDisplayScene)

        self.edgeTable = QTableWidget()
        self.edgeTable.setColumnCount(3)
        self.edgeTable.setHorizontalHeaderLabels(["Node 1", "Node 2", "Weight"])
        self.edgeTable.cellChanged.connect(self.updateEdgeWeight)

        self.edgeStartInput = QLineEdit()
        self.edgeEndInput = QLineEdit()
        self.edgeWeightInput = QLineEdit()
        self.addEdgeButton = QPushButton("Add Edge")
        self.addEdgeButton.clicked.connect(self.addEdgeManually)

        edgeInputLayout = QHBoxLayout()
        edgeInputLayout.addWidget(QLabel("Node 1:"))
        edgeInputLayout.addWidget(self.edgeStartInput)
        edgeInputLayout.addWidget(QLabel("Node 2:"))
        edgeInputLayout.addWidget(self.edgeEndInput)
        edgeInputLayout.addWidget(QLabel("Weight:"))
        edgeInputLayout.addWidget(self.edgeWeightInput)
        edgeInputLayout.addWidget(self.addEdgeButton)

        leftLayout.addWidget(QLabel("Graph"))
        leftLayout.addWidget(self.graphDisplayView)
        leftLayout.addWidget(QLabel("Optimal Path"))
        leftLayout.addWidget(self.solutionDisplayView)

        rightLayout.addWidget(QLabel("Edges"))
        rightLayout.addWidget(self.edgeTable)
        rightLayout.addLayout(edgeInputLayout)
        rightLayout.addWidget(QLabel("Calculated Path"))
        rightLayout.addWidget(self.resultDisplay)
        rightLayout.addWidget(self.useModificationCheckBox)
        rightLayout.addWidget(self.calculateRouteButton)
        rightLayout.addWidget(self.undoLastActionButton)
        rightLayout.addWidget(self.clearGraphButton)

        mainLayout.addLayout(leftLayout, 1)
        mainLayout.addLayout(rightLayout, 2)

        self.centralWidget.setLayout(mainLayout)

        self.nodes = []
        self.edges = []
        self.nodePositions = {}
        self.selectedNode = None
        self.history = []

    def addEdgeManually(self):
        try:
            node1 = int(self.edgeStartInput.text())
            node2 = int(self.edgeEndInput.text())
            weight = float(self.edgeWeightInput.text())

            if node1 not in self.nodes or node2 not in self.nodes:
                self.resultDisplay.setText("One of the nodes does not exist.")
                return

            self.edges.append([node1, node2, weight])
            self.history.append(("edge", node1, node2))
            self.redrawGraph()
        except ValueError:
            self.resultDisplay.setText("Invalid input.")

    def handleGraphClick(self, event):
        if event.button() == Qt.LeftButton:
            scenePos = self.graphDisplayView.mapToScene(event.pos())
            nodeId = len(self.nodes)
            self.nodes.append(nodeId)
            self.nodePositions[nodeId] = (scenePos.x(), scenePos.y())
            self.history.append(("node", nodeId))
            self.redrawGraph()

    def clearGraph(self):
        self.graphDisplayScene.clear()
        self.solutionDisplayScene.clear()
        self.edgeTable.setRowCount(0)
        self.nodes = []
        self.edges = []
        self.nodePositions = {}
        self.selectedNode = None
        self.history = []
        self.resultDisplay.clear()

    def drawArrow(self, scene, x1, y1, x2, y2, pen, nodeRadius=10):
        angle = np.arctan2(y2 - y1, x2 - x1)

        x1Adj = x1 + nodeRadius * np.cos(angle)
        y1Adj = y1 + nodeRadius * np.sin(angle)
        x2Adj = x2 - nodeRadius * np.cos(angle)
        y2Adj = y2 - nodeRadius * np.sin(angle)

        scene.addLine(x1Adj, y1Adj, x2Adj, y2Adj, pen)

        arrowSize = 10
        arrowP1 = QPointF(x2Adj - arrowSize * np.cos(angle - np.pi / 6),
                          y2Adj - arrowSize * np.sin(angle - np.pi / 6))
        arrowP2 = QPointF(x2Adj - arrowSize * np.cos(angle + np.pi / 6),
                          y2Adj - arrowSize * np.sin(angle + np.pi / 6))

        scene.addPolygon(QPolygonF([QPointF(x2Adj, y2Adj), arrowP1, arrowP2]), pen, QBrush(pen.color()))

    def drawOptimalPath(self, path):
        self.solutionDisplayScene.clear()
        pen = QPen(Qt.green, 2)
        brush = QBrush(Qt.red)

        for i, (x, y) in self.nodePositions.items():
            self.solutionDisplayScene.addEllipse(x - 10, y - 10, 20, 20, pen, brush)
            text = self.solutionDisplayScene.addText(str(i))
            text.setPos(x - text.boundingRect().width() / 2, y - text.boundingRect().height() / 2)

        for i in range(len(path) - 1):
            x1, y1 = self.nodePositions[path[i]]
            x2, y2 = self.nodePositions[path[i + 1]]
            self.drawArrow(self.solutionDisplayScene, x1, y1, x2, y2, pen)

    def getEdgeWeight(self, i, j):
        weights = [edge[2] for edge in self.edges if (edge[0] == i and edge[1] == j)]
        return min(weights, default=float('inf'))

    def lockTableColumns(self):
        for row in range(self.edgeTable.rowCount()):
            for col in (0, 1):
                item = self.edgeTable.item(row, col)
                if item is not None:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

    def redrawGraph(self):
        self.graphDisplayScene.clear()
        self.edgeTable.setRowCount(0)

        for node, (x, y) in self.nodePositions.items():
            brush = QBrush(Qt.red)
            pen = QPen(Qt.black)
            self.graphDisplayScene.addEllipse(x - 10, y - 10, 20, 20, pen, brush)
            text = self.graphDisplayScene.addText(str(node))
            text.setPos(x - text.boundingRect().width() / 2, y - text.boundingRect().height() / 2)

        for node1, node2, weight in self.edges:
            x1, y1 = self.nodePositions[node1]
            x2, y2 = self.nodePositions[node2]
            pen = QPen(Qt.blue, 2)
            self.drawArrow(self.graphDisplayScene, x1, y1, x2, y2, pen)

            row = self.edgeTable.rowCount()
            self.edgeTable.insertRow(row)
            self.edgeTable.setItem(row, 0, QTableWidgetItem(str(node1)))
            self.edgeTable.setItem(row, 1, QTableWidgetItem(str(node2)))
            self.edgeTable.setItem(row, 2, QTableWidgetItem(str(weight)))
            self.lockTableColumns()

    def find_hamiltonian_cycle(self):
        n = len(self.nodes)
        dist_matrix = np.full((n, n), float('inf'))

        for edge in self.edges:
            i, j, weight = edge
            dist_matrix[i][j] = weight
        if n == 0:
            return []

        # Выбираем случайную стартовую вершину
        start_node = random.randint(0, n - 1)
        path = [start_node]
        visited = set(path)

        def dfs(current):
            if len(path) == n:
                # Проверяем, есть ли обратное ребро в стартовую вершину
                if dist_matrix[current][start_node] != float('inf'):
                    return True
                return False

            # Сортируем соседей случайным образом для разнообразия
            neighbors = list(range(n))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited and dist_matrix[current][neighbor] != float('inf'):
                    visited.add(neighbor)
                    path.append(neighbor)

                    if dfs(neighbor):
                        return True

                    # Backtrack
                    path.pop()
                    visited.remove(neighbor)

            return False

        if dfs(start_node):
            return path
        else:
            return []

    def totalDistance(self, path1):
        path = path1.copy()
        path.append(path[0])
        return sum(self.getEdgeWeight(path[i], path[i + 1]) for i in range(len(path) - 1))


    def get_neighbor_path(self, path):
        if len(path) < 2:
            return path.copy()

        # Выбираем два различных индекса (кроме последнего, чтобы сохранить цикл)
        i, j = sorted(random.sample(range(len(path) - 1), 2))
        neighbor = path.copy()
        # Переворачиваем участок пути между i и j
        neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
        return neighbor

    def calculateOptimalRoute(self):
        self.edges = [[0, 1, 3.0], [1, 0, 3.0], [5, 0, 3.0], [5, 1, 3.0], [1, 5, 3.0], [5, 4, 4.0], [5, 3, 5.0], [5, 2, 3.0], [1, 2, 8.0], [2, 1, 3.0], [2, 3, 1.0], [3, 2, 8.0], [3, 4, 1.0], [4, 3, 3.0], [4, 0, 3.0], [0, 4, 1.0]]
        self.nodes = [0, 1, 2, 3, 4, 5]
        self.nodePositions = {0: (29.0, -52.0), 1: (-45.0, 26.0), 2: (23.0, 74.0), 3: (84.0, 49.0), 4: (95.0, -8.0), 5: (29.0, 20.0)}
        if not self.edges:
            return

        t_1 = time.perf_counter()

        currentPath = self.find_hamiltonian_cycle()
        currentDistance = self.totalDistance(currentPath)
        firstDistance = currentDistance
        print(currentPath, currentDistance)

        bestPath = currentPath[:]
        bestDistance = currentDistance

        temperature = 1000
        start_temp = temperature
        coolingRate = 0.999
        minTemperature = 0.001
        iterationsPerTemp = 1
        iteration = 0

        while temperature > minTemperature:
            if self.useModificationCheckBox.isChecked():
                temperature = start_temp / (1 + iteration)
            else:
                temperature = start_temp * (coolingRate ** iteration)

            for _ in range(iterationsPerTemp):
                newPath = self.get_neighbor_path(currentPath)
                newDistance = self.totalDistance(newPath)
                print(newDistance)
                delta = newDistance - currentDistance

                if delta <= 0:
                    currentPath = newPath
                    currentDistance = newDistance

                    bestPath = newPath
                    bestDistance = newDistance
                    # print(f"{iteration} New best distance: {bestDistance}")

                elif random.random() < math.exp(-(delta) / temperature):
                    currentPath = newPath
                    currentDistance = newDistance
                    # print(f"{iteration} New distance: {newDistance}")

            iteration += 1

        print(f"Iteration and first distance: {iteration} {firstDistance}")
        if bestPath:
            bestPath.append(bestPath[0])
            t_2 = time.perf_counter()
            dt = t_2 - t_1
            self.resultDisplay.setText(f"Время выполнения: {dt:.4f} сек")
            self.resultDisplay.append(f"Optimal Path: {' -> '.join(map(str, bestPath))}")
            self.resultDisplay.append(f"Path Length: {bestDistance:.4f}")
            print(bestPath)
            self.drawOptimalPath(bestPath)
        else:
            self.resultDisplay.setText("No path found.")

    def undoLastAction(self):
        if not self.history:
            return

        lastAction = self.history.pop()

        if lastAction[0] == "node":
            nodeId = lastAction[1]
            del self.nodePositions[nodeId]
            self.nodes.remove(nodeId)
        elif lastAction[0] == "edge":
            node1, node2 = lastAction[1], lastAction[2]
            self.edges = [edge for edge in self.edges if not (edge[0] == node1 and edge[1] == node2)]

        self.redrawGraph()

    def updateEdgeWeight(self, row, col):
        if col == 2:
            try:
                newWeight = float(self.edgeTable.item(row, col).text())
                self.edges[row][2] = newWeight
            except ValueError:
                pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TravelingSalesmanApp()
    window.show()
    sys.exit(app.exec_())