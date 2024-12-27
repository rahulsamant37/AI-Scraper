graph TD
    A[User Input Fields] -->|Example Input| B["Fields = ['price', 'title', 'description']"]
    
    subgraph Dynamic_Listing_Model[Dynamic Listing Model Creation]
        B --> C[Create Single Item Structure]
        C -->|Creates| D[Pydantic Model]
        D --> E["Single Item Schema:
        {
            'price': string,
            'title': string,
            'description': string
        }"]
    end
    
    subgraph Container_Model[Container Model Creation]
        E --> F[Create Container Structure]
        F -->|Wraps Items| G["Final Schema:
        {
            'listings': [
                {item1},
                {item2},
                {item3},
                ...
            ]
        }"]
    end
    
    H[Real World Example] --> I["User wants to scrape:
    - Product Name
    - Price
    - Rating"]
    
    I --> J["Creates Model:
    {
        'listings': [
            {
                'Product Name': 'iPhone 13',
                'Price': '$799',
                'Rating': '4.5'
            },
            {
                'Product Name': 'Galaxy S21',
                'Price': '$699',
                'Rating': '4.3'
            }
        ]
    }"]
    
    style Dynamic_Listing_Model fill:#ffd,stroke:#333
    style Container_Model fill:#dff,stroke:#333








# Without Dynamic Model:
data = {"random_field": "value"}  # No structure, could be anything

# With Dynamic Model:
data = {
    "price": "100",
    "title": "Product",
    "description": "Details"
}  # Structured, validated data