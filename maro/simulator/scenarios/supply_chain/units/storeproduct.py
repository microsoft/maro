
from .product import ProductUnit


class StoreProductUnit(ProductUnit):
    def get_sale_mean(self):
        return self.seller.sale_mean()

    def get_sale_std(self):
        return self.seller.sale_std()

    def get_selling_price(self):
        return self.facility.skus[self.product_id].price
