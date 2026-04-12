import React, { useState, useMemo } from 'react';
import { Edit2, Save, X, Plus } from 'lucide-react';
import { getBills, getProducts } from '../storage';
import '../styles/StoreDisplayPlanner.css';

interface DisplayZone {
  id: string;
  name: string;
  color: string;
  description: string;
  productIds: string[];
  isEditing: boolean;
  type: 'hot' | 'impulse' | 'bundle' | 'seasonal' | 'accessory';
}

const DEFAULT_ZONES: DisplayZone[] = [
  {
    id: 'zone-1',
    name: 'Hot Items (Eye Level)',
    color: '#ff6b6b',
    description: 'Top selling products - Best visibility',
    productIds: [],
    isEditing: false,
    type: 'hot',
  },
  {
    id: 'zone-2',
    name: 'New Arrivals',
    color: '#4ecdc4',
    description: 'New products - Customizable placement',
    productIds: [],
    isEditing: false,
    type: 'seasonal',
  },
  {
    id: 'zone-3',
    name: 'Impulse Buy (Checkout)',
    color: '#ffe66d',
    description: 'Small, quick-pick items at counter',
    productIds: [],
    isEditing: false,
    type: 'impulse',
  },
  {
    id: 'zone-4',
    name: 'Combo Bundles',
    color: '#95e1d3',
    description: 'Complementary products together',
    productIds: [],
    isEditing: false,
    type: 'bundle',
  },
  {
    id: 'zone-5',
    name: 'Accessories & Support',
    color: '#a8e6cf',
    description: 'Supporting products',
    productIds: [],
    isEditing: false,
    type: 'accessory',
  },
];

export default function StoreDisplayPlanner() {
  const [zones, setZones] = useState<DisplayZone[]>(DEFAULT_ZONES);
  const allProducts = getProducts();
  const bills = getBills();

  // Calculate sales data
  const salesData = useMemo(() => {
    const data: { [key: string]: { quantity: number; revenue: number } } = {};

    bills.forEach(bill => {
      bill.items.forEach(item => {
        if (!data[item.productId]) {
          data[item.productId] = { quantity: 0, revenue: 0 };
        }
        data[item.productId].quantity += item.quantity;
        data[item.productId].revenue += item.price * item.quantity;
      });
    });

    return data;
  }, [bills]);

  // Get top selling products
  const topProducts = useMemo(() => {
    return allProducts
      .map(p => ({
        ...p,
        quantity: salesData[p.id]?.quantity || 0,
        revenue: salesData[p.id]?.revenue || 0,
      }))
      .sort((a, b) => b.revenue - a.revenue);
  }, [allProducts, salesData]);

  // Auto-populate zones on first load
  React.useEffect(() => {
    const hasProducts = zones.some(z => z.productIds.length > 0);
    if (!hasProducts && topProducts.length > 0) {
      const newZones = [...zones];

      // Hot items - top 3
      newZones[0].productIds = topProducts.slice(0, 3).map(p => p.id);

      // New arrivals - items with no sales
      const noSalesProducts = allProducts.filter(p => !salesData[p.id]).slice(0, 2);
      newZones[1].productIds = noSalesProducts.map(p => p.id);

      // Impulse - small items
      newZones[2].productIds = topProducts.slice(3, 5).map(p => p.id);

      setZones(newZones);
    }
  }, []);

  const toggleZoneEdit = (zoneId: string) => {
    setZones(zones.map(z =>
      z.id === zoneId ? { ...z, isEditing: !z.isEditing } : z
    ));
  };

  const addProductToZone = (zoneId: string, productId: string) => {
    setZones(zones.map(z => {
      if (z.id === zoneId && !z.productIds.includes(productId)) {
        return { ...z, productIds: [...z.productIds, productId] };
      }
      return z;
    }));
  };

  const removeProductFromZone = (zoneId: string, productId: string) => {
    setZones(zones.map(z =>
      z.id === zoneId
        ? { ...z, productIds: z.productIds.filter(pid => pid !== productId) }
        : z
    ));
  };

  const getProductName = (productId: string) => {
    return allProducts.find(p => p.id === productId)?.name || 'Unknown';
  };

  const getProductSales = (productId: string) => {
    return salesData[productId] || { quantity: 0, revenue: 0 };
  };

  return (
    <div className="store-display-planner">
      <h2 className="card-title">Store Display Planner</h2>
      <p className="planner-subtitle">Optimize your store layout based on sales data. Customize zones as needed.</p>

      {/* Store Layout Visualization */}
      <div className="card store-layout">
        <h3 className="card-title" style={{ marginTop: 0 }}>📍 Store Layout</h3>
        <div className="store-visualization">
          <div className="entrance">ENTRANCE / COUNTER</div>

          <div className="zones-grid">
            {zones.map(zone => (
              <div
                key={zone.id}
                className="zone"
                style={{ backgroundColor: zone.color + '20', borderColor: zone.color }}
              >
                <div className="zone-header" style={{ backgroundColor: zone.color }}>
                  <div>
                    <h4>{zone.name}</h4>
                    <p className="zone-desc">{zone.description}</p>
                  </div>
                  <button
                    className="btn-edit"
                    onClick={() => toggleZoneEdit(zone.id)}
                  >
                    {zone.isEditing ? <X size={16} /> : <Edit2 size={16} />}
                  </button>
                </div>

                <div className="zone-products">
                  {zone.productIds.length === 0 ? (
                    <p className="empty-zone">No products assigned</p>
                  ) : (
                    zone.productIds.map(productId => {
                      const sales = getProductSales(productId);
                      return (
                        <div key={productId} className="zone-product">
                          <div className="product-info">
                            <strong>{getProductName(productId)}</strong>
                            <small>
                              {sales.quantity > 0 ? `${sales.quantity} sold` : 'New arrival'}
                            </small>
                          </div>
                          {zone.isEditing && (
                            <button
                              className="btn-remove"
                              onClick={() => removeProductFromZone(zone.id, productId)}
                            >
                              ✕
                            </button>
                          )}
                        </div>
                      );
                    })
                  )}
                </div>

                {zone.isEditing && (
                  <div className="zone-edit">
                    <select
                      className="form-select"
                      onChange={e => {
                        if (e.target.value) {
                          addProductToZone(zone.id, e.target.value);
                          e.target.value = '';
                        }
                      }}
                      defaultValue=""
                    >
                      <option value="">+ Add product...</option>
                      {allProducts
                        .filter(p => !zone.productIds.includes(p.id))
                        .map(p => (
                          <option key={p.id} value={p.id}>
                            {p.name}
                          </option>
                        ))}
                    </select>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sales Analytics */}
      <div className="card">
        <h3 className="card-title">📊 Top Selling Products</h3>
        {topProducts.length === 0 ? (
          <p className="empty-state">No sales data yet. Create some bills first!</p>
        ) : (
          <div className="sales-table">
            <div className="table-header">
              <div className="col-product">Product</div>
              <div className="col-quantity">Qty Sold</div>
              <div className="col-revenue">Revenue</div>
              <div className="col-margin">Recommended Zone</div>
            </div>
            {topProducts.map((product, index) => {
              let recommendedZone = 'Accessories';
              if (index < 3) recommendedZone = 'Hot Items';
              else if (index < 5) recommendedZone = 'Impulse Buy';

              return (
                <div key={product.id} className="table-row">
                  <div className="col-product">
                    <strong>{product.name}</strong>
                  </div>
                  <div className="col-quantity">{product.quantity}</div>
                  <div className="col-revenue">₹{product.revenue.toLocaleString()}</div>
                  <div className="col-margin">{recommendedZone}</div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* New Arrivals (No Sales Yet) */}
      <div className="card">
        <h3 className="card-title">✨ New Arrivals (No Sales Yet)</h3>
        {allProducts.filter(p => !salesData[p.id]).length === 0 ? (
          <p className="empty-state">All products have sales data</p>
        ) : (
          <div className="new-arrivals-list">
            {allProducts
              .filter(p => !salesData[p.id])
              .map(product => (
                <div key={product.id} className="arrival-item">
                  <div className="arrival-info">
                    <strong>{product.name}</strong>
                    <small>{product.category}</small>
                  </div>
                  <span className="badge-new">NEW</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
