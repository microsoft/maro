
from .product import ProductUnit


class StoreProductUnit(ProductUnit):
    def get_latest_sale(self):
        sale_hist = self.seller.config.get("sale_hist", 0)

        # TODO: why demand not sold?
        return 0 if sale_hist == 0 else self.seller.demand

    def get_sale_mean(self):
        return self.seller.sale_mean()

    def get_sale_std(self):
        return self.seller.sale_std()

    def get_selling_price(self):
        return self.facility.skus[self.product_id].price
