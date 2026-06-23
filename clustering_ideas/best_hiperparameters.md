# ERROR = 0.5xMSE_TIME + 0.5xMSE_SPACE
## Z wykorzystaniem Shortest Path

### K-Prototypes

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 11 | 0.8 | 0.2 |
| **Saint Arnoult** | 11 | 0.8 | 0.2 |
| **Ingolstadt** | 9 | 0.8 | 0.2 |

### Similarity Measure (K-Medoids)

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 11 | 0.2 | 0.8 |
| **Saint Arnoult** | 13 | 0.2 | 0.8 |
| **Ingolstadt** | 19 | 0.7 | 0.3 |

---

## Bez odległości Shortest Path (Euklidesowe)

### K-Prototypes

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 11 | 0.8 | 0.2 |
| **Saint Arnoult** | 11 | 0.8 | 0.2 |
| **Ingolstadt** | 11 | 0.8 | 0.2 |

### Similarity Measure (K-Medoids)

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 11 | 0.3 | 0.7 |
| **Saint Arnoult** | 13 | 0.3 | 0.7 |
| **Ingolstadt** | 19 | 0.9 | 0.1 |

### Spatial (Przestrzenne)

| Miasto | Początkowe `K_initial` | Ostateczna liczba klastrów (po fuzji) |
| --- | --- | --- |
| **Provins** | 12 | 6 |
| **Saint Arnoult** | 12 | 6 |
| **Ingolstadt** | 8/12 | 1/5 |

### Spatiotemporal (Czasoprzestrzenne)

| Miasto | Początkowe `K_initial` | Ostateczna liczba klastrów (po fuzji) |
| --- | --- | --- |
| **Provins** | 12 | 8 |
| **Saint Arnoult** | 10 | 10 |
| **Ingolstadt** | 12 | 11 |

# ERROR = TIME_WEIGHTxMSE_TIME + SPACE_WEIGHTxMSE_SPACE
## Z wykorzystaniem Shortest Path

### K-Prototypes

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 11 | 1.0 | 0.0 |
| **Saint Arnoult** | 11 | 1.0 | 0.0 |
| **Ingolstadt** | 11 | 1.0 | 0.0 |

### Similarity Measure (K-Medoids)

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 13 | 1.0 | 0.0 |
| **Saint Arnoult** | 11 | 1.0 | 0.0 |
| **Ingolstadt** | 9 | 1.0 | 0.0 |

---

## Bez odległości Shortest Path (Euklidesowe)

### K-Prototypes

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 11 | 1.0 | 0.0 |
| **Saint Arnoult** | 11 | 1.0 | 0.0 |
| **Ingolstadt** | 11 | 1.0 | 0.0 |

### Similarity Measure (K-Medoids)

| Miasto | `num_clusters` (K) | `TIME_WEIGHT` | `SPACE_WEIGHT` |
| --- | --- | --- | --- |
| **Provins** | 13 | 1.0 | 0.0 |
| **Saint Arnoult** | 11 | 1.0 | 0.0 |
| **Ingolstadt** | 9 | 1.0 | 1.0 |

### Spatial (Przestrzenne)

| Miasto | Początkowe `K_initial` | Ostateczna liczba klastrów (po fuzji) |
| --- | --- | --- |
| **Provins** |  |  |
| **Saint Arnoult** |  |  |
| **Ingolstadt** |  |  |

### Spatiotemporal (Czasoprzestrzenne)

| Miasto | Początkowe `K_initial` | Ostateczna liczba klastrów (po fuzji) |
| --- | --- | --- |
| **Provins** |  |  |
| **Saint Arnoult** |  |  |
| **Ingolstadt** |  |  |
