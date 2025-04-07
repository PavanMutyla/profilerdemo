from typing import Dict, List

from RAG.entities import KBCategory
CATEGORY_DB_MAPPING: Dict[KBCategory, List[str]] = {
    KBCategory.ProductCategory: ["fin_prods_retriever"],
    KBCategory.InvestmentRegulations: ["regulations_retriever"],
    KBCategory.TaxationDetails: ["tax_retriever", "general_retriever"],  # Fetch from two retrievers
    KBCategory.General: ["general_retriever"],
    KBCategory.CulturalAspects: ["tailvy_api"],
    KBCategory.MarketSegments: ["tailvy_api"]
}
